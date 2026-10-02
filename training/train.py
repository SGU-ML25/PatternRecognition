"""Huấn luyện PoseNet (SimpleBaseline) trên COCO-Keypoints crops."""
import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CocoPoseCrops
from evaluate import coco_ap, pck, predict
from model import PoseNet

ROOT = Path(__file__).resolve().parents[1]


def weighted_mse(pred, target, w):
    return (((pred - target) ** 2).mean((2, 3)) * w).mean() * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="chỉ dùng N mẫu (chạy thử)")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--resume", action="store_true", help="train tiếp từ <out>/last.pt")
    ap.add_argument("--resume-from", default=None, help="train tiếp từ một last.pt ở nơi khác")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = CocoPoseCrops("train", train=True)
    val_ds = CocoPoseCrops("val", train=False)
    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
    if args.eval_limit:  # chạy thử: đánh giá trên tập con (AP chỉ mang tính tham khảo)
        for k in ("ann_id", "image_id", "kps", "center", "side", "area"):
            setattr(val_ds, k, getattr(val_ds, k)[:args.eval_limit])
    dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

    model = PoseNet(args.backbone).to(device).to(memory_format=torch.channels_last)
    core = model  # model gốc (để lưu state_dict không có tiền tố "module.")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel trên {torch.cuda.device_count()} GPU")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total, warm = args.epochs * len(dl), min(1000, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()
    start, best = 1, -1.0
    log_path = out / "train_log.csv"
    resume_ck = Path(args.resume_from) if args.resume_from else out / "last.pt"
    if (args.resume or args.resume_from) and resume_ck.exists():
        ck = torch.load(resume_ck, map_location="cpu", weights_only=False)
        assert ck["backbone"] == args.backbone, f"checkpoint là {ck['backbone']}"
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"Resume từ epoch {ck['epoch']} ({resume_ck}), best AP={best:.4f}")
        old_log = resume_ck.parent / "train_log.csv"
        if old_log.exists() and not log_path.exists():
            shutil.copy(old_log, log_path)  # giữ lịch sử các epoch trước

    if start == 1 or not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "lr", "AP", "AP50", "AP75", "AR",
                                    "PCK@0.1_control", "minutes"])
    print(f"{args.backbone}: {len(train_ds)} mẫu train, {len(dl)} iter/epoch, {len(val_ds)} mẫu val")

    for ep in range(start, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for it, (x, hm, w, _, _) in enumerate(dl):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            hm, w = hm.to(device, non_blocking=True), w.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = weighted_mse(pred.float(), hm, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); n += 1
            if it % 200 == 0:
                print(f"ep {ep} it {it}/{len(dl)} loss {run / n:.6f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(it + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)
        xy, conf = predict(model, val_ds, device, flip=True)
        coco, pk = coco_ap(val_ds, xy, conf), pck(val_ds, xy)
        mins = (time.time() - t0) / 60
        print(f"== ep {ep}: loss {run / n:.6f} AP {coco['AP']:.4f} AP50 {coco['AP50']:.4f} "
              f"PCK@0.1(control) {pk['PCK@0.1']['control_joints']:.4f} ({mins:.1f} phút)", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, run / n, sched.get_last_lr()[0], coco["AP"], coco["AP50"],
                                    coco["AP75"], coco["AR"], pk["PCK@0.1"]["control_joints"], mins])
        ck = {"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": max(best, coco["AP"]),
              "backbone": args.backbone, "coco": coco}
        torch.save(ck, out / "last.pt")
        if coco["AP"] > best:
            best = coco["AP"]
            torch.save({"model": core.state_dict(), "backbone": args.backbone, "epoch": ep, "coco": coco},
                       out / "best.pt")
            print(f"   -> best.pt (AP {best:.4f})")


if __name__ == "__main__":
    main()
