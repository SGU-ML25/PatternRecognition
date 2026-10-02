"""Sinh biểu đồ + ảnh minh hoạ cho báo cáo từ log train và kết quả đánh giá.

Đầu vào:  logs/train_<name>.log (stdout của train.py), checkpoints*/train_log.csv,
          report/eval_<name>.json (evaluate.py)
Đầu ra:   report/figures/*.png
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
# Bảng màu categorical (đã kiểm định CVD), chữ dùng màu mực trung tính
C1, C2, C3, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def parse_iter_log(path):
    """Đọc dòng 'ep E it I/N loss L lr X S img/s' -> (epoch thực, loss trung bình chạy)."""
    xs, ys = [], []
    pat = re.compile(r"ep (\d+) it (\d+)/(\d+) loss ([\d.e-]+)")
    for line in open(path):
        m = pat.search(line)
        if m and int(m.group(2)) > 0:
            e, i, n, loss = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            xs.append(e - 1 + i / n)
            ys.append(loss)
    return np.array(xs), np.array(ys)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("  ->", FIG / name)


def fig_loss(runs):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, log, _), c in zip(runs, [C1, C2, C3]):
        if log and Path(log).exists():
            x, y = parse_iter_log(log)
            ax.plot(x, y * 1e4, color=c, label=name)
            if len(x):
                ax.annotate(name, (x[-1], y[-1] * 1e4), xytext=(4, 0), textcoords="offset points",
                            color=INK2, va="center", fontsize=9)
    ax.set_title("Loss huấn luyện (MSE heatmap, trung bình chạy trong epoch)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss × 10⁻⁴")
    if len(runs) > 1:
        ax.legend(loc="upper right")
    save(fig, "train_loss.png")


def fig_ap(runs):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, _, csv_path), c in zip(runs, [C1, C2, C3]):
        rows = read_csv(csv_path)
        ep = [r["epoch"] for r in rows]
        ap = [r["AP"] * 100 for r in rows]
        ax.plot(ep, ap, color=c, marker="o", markersize=4, label=name)
        b = int(np.argmax(ap))
        ax.annotate(f"{name}: {ap[b]:.1f}", (ep[b], ap[b]), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.set_title("COCO Keypoint AP trên val2017 theo epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("AP (%)")
    if len(runs) > 1:
        ax.legend(loc="lower right")
    save(fig, "val_ap.png")


def fig_ap_breakdown(csv_path, name):
    rows = read_csv(csv_path)
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, c in [("AP50", C1), ("AP75", C2), ("AP", C3)]:
        v = [r[key] * 100 for r in rows]
        ax.plot(ep, v, color=c, label=key)
        ax.annotate(key, (ep[-1], v[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.set_title(f"{name}: AP ở các ngưỡng OKS")
    ax.set_xlabel("epoch"); ax.set_ylabel("%"); ax.legend(loc="lower right")
    save(fig, "val_ap_breakdown.png")


def fig_lr(csv_path):
    rows = read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot([r["epoch"] for r in rows], [r["lr"] for r in rows], color=C1)
    ax.set_title("Learning rate cuối mỗi epoch (cosine decay)")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr")
    save(fig, "lr.png")


def fig_pck(ev):
    per = ev["pck"]["PCK@0.1"]["per_joint"]
    names = list(per)
    control = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip"}
    vals = [per[n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))[::-1]
    colors = [C1 if n in control else MUTED for n in names]
    ax.barh(y, vals, color=colors, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, names); ax.set_xlim(0, 105); ax.grid(axis="y", visible=False)
    ax.set_title("PCK@0.1 theo từng khớp (val2017)")
    ax.set_xlabel("% khớp dự đoán đúng (sai số < 10% kích thước người)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C1, label="khớp dùng để điều khiển"), Patch(color=MUTED, label="khớp khác")],
              loc="lower right")
    save(fig, "pck_per_joint.png")


def fig_pck_curve(ev):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    alphas = sorted(float(k.split("@")[1]) for k in ev["pck"])
    for key, c, lab in [("all", C2, "17 khớp"), ("control_joints", C1, "8 khớp điều khiển")]:
        v = [ev["pck"][f"PCK@{a}"][key] * 100 for a in alphas]
        ax.plot(alphas, v, color=c, marker="o", markersize=5, label=lab)
        for a, vi in zip(alphas, v):
            ax.annotate(f"{vi:.1f}", (a, vi), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=8, color=INK2)
    ax.set_title("PCK theo ngưỡng α"); ax.set_xlabel("α (tỉ lệ kích thước người)"); ax.set_ylabel("PCK (%)")
    ax.set_ylim(0, 105); ax.legend(loc="lower right")
    save(fig, "pck_curve.png")


def fig_examples(ckpt, val_images, n=12, seed=3):
    """Ảnh minh hoạ: dự đoán của model trên ảnh val2017."""
    import sys
    sys.path.insert(0, str(ROOT))
    from server.pose_engine import PoseEngine
    from pycocotools.coco import COCO
    from paths import ANN_DIR
    from pose_utils import SKELETON
    eng = PoseEngine(ckpt)
    coco = COCO(str(ANN_DIR / "person_keypoints_val2017.json"))
    rng = np.random.default_rng(seed)
    anns = [a for a in coco.loadAnns(coco.getAnnIds(iscrowd=False))
            if a["num_keypoints"] >= 10 and a["area"] > 20000]
    tiles = []
    for a in rng.choice(anns, n, replace=False):
        img = cv2.imread(str(Path(val_images) / coco.loadImgs(a["image_id"])[0]["file_name"]))
        x, y, w, h = a["bbox"]
        c, side = (x + w / 2, y + h / 2), max(w, h) * 1.25
        kp = eng.predict_crop(img, c, side)
        M = np.float32([[256 / side, 0, 128 - c[0] * 256 / side], [0, 256 / side, 128 - c[1] * 256 / side]])
        t = cv2.warpAffine(img, M, (256, 256))
        p = kp[:, :2] @ M[:, :2].T + M[:, 2]
        for i, j in SKELETON:
            if kp[i, 2] > 0.3 and kp[j, 2] > 0.3:
                cv2.line(t, tuple(map(int, p[i])), tuple(map(int, p[j])), (232, 224, 95), 2, cv2.LINE_AA)
        for k in range(17):
            if kp[k, 2] > 0.3:
                cv2.circle(t, tuple(map(int, p[k])), 3, (59, 195, 246), -1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(n // 4)])
    FIG.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIG / "examples_val.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  ->", FIG / "examples_val.jpg")


def main():
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name:train_stdout.log:train_log.csv (lặp lại cho nhiều model)")
    ap.add_argument("--eval", help="report/eval_*.json của model chính")
    ap.add_argument("--ckpt", help="checkpoint model chính để vẽ ảnh minh hoạ")
    ap.add_argument("--val-images", default=str(ROOT / "data" / "coco" / "val2017"))
    ap.add_argument("--fig-dir", default=str(FIG))
    args = ap.parse_args()
    FIG = Path(args.fig_dir)
    runs = [tuple(r.split(":")) for r in args.run]
    fig_loss(runs)
    fig_ap(runs)
    fig_ap_breakdown(runs[0][2], runs[0][0])
    fig_lr(runs[0][2])
    if args.eval:
        ev = json.load(open(args.eval))
        fig_pck(ev)
        fig_pck_curve(ev)
    if args.ckpt:
        fig_examples(args.ckpt, args.val_images)


if __name__ == "__main__":
    main()
