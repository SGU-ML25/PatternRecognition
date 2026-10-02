"""Đánh giá model trên COCO val2017: COCO Keypoint AP (GT bbox) + PCK thân trên + tốc độ."""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import CROP, CocoPoseCrops
from model import PoseNet
from pose_utils import FLIP_INDEX, KEYPOINTS, decode_heatmaps, heatmap_to_input, normalize_batch

from paths import ANN_DIR, ROOT

ANN_VAL = ANN_DIR / "person_keypoints_val2017.json"
CONTROL_KPTS = [5, 6, 7, 8, 9, 10, 11, 12]  # vai, khuỷu, cổ tay, hông — dùng để điều khiển


@torch.no_grad()
def predict(model, ds, device, flip=True, bs=64, workers=8):
    """Trả về keypoint trong toạ độ ảnh gốc (N,17,2) và độ tin cậy (N,17)."""
    model.eval()
    dl = DataLoader(ds, bs, shuffle=False, num_workers=workers, pin_memory=True)
    all_xy, all_conf = [], []
    for x, _, _, M, idx in dl:
        x = normalize_batch(x.to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            hm = model(x).float()
            if flip:
                hf = model(torch.flip(x, [3])).float()
                hf = torch.flip(hf, [3])[:, FLIP_INDEX]
                hf[:, :, :, 1:] = hf[:, :, :, :-1].clone()  # bù lệch 1 pixel khi lật
                hm = (hm + hf) / 2
        coords, conf = decode_heatmaps(hm.cpu().numpy())
        xy_in = heatmap_to_input(coords)
        for b in range(len(idx)):
            i = int(idx[b])
            Minv = np.linalg.inv(np.vstack([M[b].numpy().astype(np.float64), [0, 0, 1]]))[:2]
            xy_crop = xy_in[b] @ Minv[:, :2].T + Minv[:, 2]
            xy = (xy_crop - CROP / 2) * (ds.side[i] / CROP) + ds.center[i]
            all_xy.append(xy)
            all_conf.append(conf[b])
    return np.stack(all_xy), np.stack(all_conf)


def coco_ap(ds, xy, conf):
    results = []
    for i in range(len(ds)):
        c = conf[i]
        score = float(c[c > 0.2].mean()) if (c > 0.2).any() else 0.0
        kp = np.concatenate([xy[i], c[:, None]], 1).reshape(-1)
        results.append({"image_id": int(ds.image_id[i]), "category_id": 1,
                        "keypoints": kp.round(2).tolist(), "score": score})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN_VAL))
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.useSegm = None
        ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L"]
    return dict(zip(names, [float(s) for s in ev.stats]))


def pck(ds, xy, alphas=(0.05, 0.1, 0.2)):
    """PCK: khớp đúng nếu sai số < alpha * max(w,h) của bbox. Tính riêng từng khớp."""
    gt_xy, vis = [], []
    for i in range(len(ds)):
        k = ds.kps[i]
        gt_xy.append((k[:, :2] - CROP / 2) * (ds.side[i] / CROP) + ds.center[i])
        vis.append(k[:, 2] > 0)
    gt_xy, vis = np.stack(gt_xy), np.stack(vis)
    size = (ds.side / 1.5)[:, None]
    err = np.linalg.norm(xy - gt_xy, axis=-1) / size
    out = {}
    for a in alphas:
        ok = (err < a) & vis
        per = ok.sum(0) / np.maximum(vis.sum(0), 1)
        out[f"PCK@{a}"] = {
            "all": float(ok.sum() / vis.sum()),
            "control_joints": float(ok[:, CONTROL_KPTS].sum() / vis[:, CONTROL_KPTS].sum()),
            "per_joint": {KEYPOINTS[j]: float(per[j]) for j in range(len(KEYPOINTS))},
        }
    return out


@torch.no_grad()
def speed(model, device, n=200):
    model.eval()
    x = torch.randn(1, 3, 256, 256, device=device)
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "resnet18" / "best.pt"))
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "report" / "eval_results.json"))
    args = ap.parse_args()
    device = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PoseNet(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).to(memory_format=torch.channels_last)
    ds = CocoPoseCrops("val", train=False)
    xy, conf = predict(model, ds, device, flip=not args.no_flip)
    res = {"checkpoint": args.ckpt, "backbone": ck["backbone"], "epoch": ck.get("epoch"),
           "flip_test": not args.no_flip, "num_instances": len(ds),
           "coco": coco_ap(ds, xy, conf), "pck": pck(ds, xy),
           "latency_ms_fp16_bs1": speed(model, device)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "pck"}, indent=2))
    for a, v in res["pck"].items():
        print(a, "all=%.3f control=%.3f" % (v["all"], v["control_joints"]))


if __name__ == "__main__":
    main()
