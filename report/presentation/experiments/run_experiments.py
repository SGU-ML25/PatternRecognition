"""Thí nghiệm so sánh cho presentation SkyPose. Ghi JSON ra argv[1]."""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import multiprocessing as mp
mp.set_start_method("fork", force=True)

ROOT = Path("/mnt/data/1_Projects/PatternRecognition")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))
from dataset import CocoPoseCrops  # noqa
from evaluate import coco_ap, pck, predict  # noqa
from model import PoseNet  # noqa
from pose_utils import OKS_SIGMAS  # noqa
from server.pose_engine import PoseEngine  # noqa

OUT = {}
KAGGLE = ROOT / "runs/20261002-1405_resnet18_kaggle/resnet18/best.pt"
LOCAL = ROOT / "checkpoints/resnet18/best.pt"


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = PoseNet(ck["backbone"], pretrained=False); m.load_state_dict(ck["model"])
    return m.cuda().to(memory_format=torch.channels_last)


# A. COCO val (GT bbox): flip on/off, epoch 3 vs 20
ds = CocoPoseCrops("val", train=False)
A = {}
for name, p in [("ep20", KAGGLE), ("ep3", LOCAL)]:
    m = load(p)
    for flip in (True, False):
        if name == "ep3" and not flip:
            continue
        t = time.time()
        xy, conf = predict(m, ds, "cuda", flip=flip, workers=6)
        c, k = coco_ap(ds, xy, conf), pck(ds, xy)
        A[f"{name}_flip{int(flip)}"] = {"AP": c["AP"], "AP50": c["AP50"], "AP75": c["AP75"], "AR": c["AR"],
                                        "PCK0.1_all": k["PCK@0.1"]["all"],
                                        "PCK0.1_ctrl": k["PCK@0.1"]["control_joints"],
                                        "PCK0.05_ctrl": k["PCK@0.05"]["control_joints"]}
        print(name, flip, A[f"{name}_flip{int(flip)}"], f"{time.time()-t:.0f}s", flush=True)
    del m
OUT["coco_gtbox"] = A

# B. Pipeline thật trên ảnh gốc (không có GT bbox): 1 lượt vs 2 lượt
coco = json.load(open(ROOT / "data/coco/annotations/person_keypoints_val2017.json"))
by = {}
for a in coco["annotations"]:
    if not a.get("iscrowd", 0):
        by.setdefault(a["image_id"], []).append(a)
fn = {im["id"]: im["file_name"] for im in coco["images"]}
single = sorted(i for i, an in by.items() if len(an) == 1 and an[0]["num_keypoints"] >= 10)
random.seed(0)
sample = random.sample(single, min(500, len(single)))


def oks(pred, gt, area):
    vis = gt[:, 2] > 0
    d2 = ((pred[:, :2] - gt[:, :2]) ** 2).sum(1)
    return float(np.exp(-d2 / (2 * (area + 1e-9) * (2 * OKS_SIGMAS) ** 2))[vis].mean())


def box_from(kp, h, w, kpts):
    good = [j for j in kpts if kp[j, 2] > 0.3]
    if len(good) < 4 or kp[[5, 6], 2].min() <= 0.3:
        return None
    lo, hi = kp[good, :2].min(0), kp[good, :2].max(0)
    side = float(np.clip((hi - lo).max() * 1.6, 0.35 * max(h, w), 1.6 * max(h, w)))
    return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, side)


B = {}
imgs = [(cv2.imread(str(ROOT / "data/coco/val2017" / fn[i])), by[i][0]) for i in sample]
for flip in (True, False):
    eng = PoseEngine(KAGGLE, flip=flip)
    res = {"1pass": [], "2pass_13": [], "2pass_17": []}
    for img, a in imgs:
        h, w = img.shape[:2]
        gt = np.array(a["keypoints"], np.float32).reshape(17, 3)
        kp1 = eng.predict_crop(img, (w / 2, h / 2), max(h, w))
        res["1pass"].append(oks(kp1, gt, a["area"]))
        for tag, kpts in (("2pass_13", range(13)), ("2pass_17", range(17))):
            b = box_from(kp1, h, w, list(kpts))
            kp = eng.predict_crop(img, b[:2], b[2]) if b else kp1
            res[tag].append(oks(kp, gt, a["area"]))
    for k, v in res.items():
        v = np.array(v)
        B[f"flip{int(flip)}_{k}"] = {"mean_oks": float(v.mean()), "oks>0.5": float((v > 0.5).mean()),
                                     "oks>0.75": float((v > 0.75).mean())}
    print(B, flush=True)
    del eng
OUT["pipeline_raw"] = {"n_images": len(imgs), **B}

# C. Tốc độ theo backbone / độ chính xác số / thiết bị (bs=1 và bs=2 cho flip-test)
C = {}


@torch.no_grad()
def bench(m, dev, half, bs, n):
    x = torch.randn(bs, 3, 256, 256, device=dev)
    if half:
        x = x.half()
    x = x.contiguous(memory_format=torch.channels_last)
    for _ in range(10):
        m(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


for bb in ("resnet18", "resnet34", "resnet50"):
    m = PoseNet(bb, pretrained=False).eval()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    r = {"params_M": params}
    mc = m.cuda().to(memory_format=torch.channels_last)
    r["gpu_fp32_bs1"] = bench(mc, "cuda", False, 1, 100)
    mh = mc.half()
    r["gpu_fp16_bs1"] = bench(mh, "cuda", True, 1, 200)
    r["gpu_fp16_bs2"] = bench(mh, "cuda", True, 2, 200)
    mcpu = PoseNet(bb, pretrained=False).eval().to(memory_format=torch.channels_last)
    r["cpu_fp32_bs1"] = bench(mcpu, "cpu", False, 1, 20)
    C[bb] = r
    print(bb, r, flush=True)
    del m, mc, mh, mcpu
    torch.cuda.empty_cache()
OUT["speed"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cpu_threads": torch.get_num_threads(), **C}

Path(sys.argv[1]).write_text(json.dumps(OUT, indent=2))
print("done")
