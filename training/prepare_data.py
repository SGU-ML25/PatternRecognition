"""Chuẩn bị dữ liệu COCO-Keypoints: cắt vùng vuông quanh mỗi người, lưu JPEG 320x320.

Nguồn ảnh (theo thứ tự ưu tiên):
- --images-dir: thư mục ảnh của split (vd. dataset COCO gắn sẵn trên Kaggle)
- val: data/coco/val2017 nếu đã tải zip
- còn lại: tải lẻ từng ảnh từ images.cocodataset.org, cắt xong không giữ ảnh gốc (tiết kiệm ổ đĩa)

Annotation đọc từ $SKYPOSE_ANN_DIR, kết quả ghi vào $SKYPOSE_CROPS (xem paths.py):
    {split}/{ann_id}.jpg + {split}_index.npz
    keypoints  (N,17,3)  toạ độ trong ảnh crop 320px, v ∈ {0,1,2}
    center     (N,2)     tâm crop trong ảnh gốc
    side       (N,)      cạnh crop trong ảnh gốc (pixel)
    ann_id, image_id, area
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from paths import ANN_DIR, CROPS_DIR, DEFAULT_VAL_IMAGES

CROP = 320
EXPAND = 1.5  # cạnh crop = 1.5 * max(w, h) của bbox


def crop_person(img, bbox):
    x, y, w, h = bbox
    c = np.array([x + w / 2, y + h / 2], np.float32)
    side = max(w, h) * EXPAND
    s = CROP / side
    M = np.array([[s, 0, CROP / 2 - s * c[0]], [0, s, CROP / 2 - s * c[1]]], np.float32)
    patch = cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return patch, c, side, M


def load_image(split, file_name, images_dir, retries=3):
    if images_dir is not None:
        return cv2.imread(str(images_dir / file_name))
    url = f"http://images.cocodataset.org/{split}2017/{file_name}"
    for _ in range(retries):
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
    return None


def process_image(split, img_info, anns, out_dir, images_dir):
    todo = [a for a in anns if not (out_dir / f"{a['id']}.jpg").exists()]
    img = load_image(split, img_info["file_name"], images_dir) if todo else None
    rows = []
    for a in anns:
        path = out_dir / f"{a['id']}.jpg"
        x, y, w, h = a["bbox"]
        c = np.array([x + w / 2, y + h / 2], np.float32)
        side = max(w, h) * EXPAND
        if not path.exists():
            if img is None:
                return []
            patch, c, side, _ = crop_person(img, a["bbox"])
            cv2.imwrite(str(path), patch, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kp = np.array(a["keypoints"], np.float32).reshape(17, 3)
        s = CROP / side
        kp[:, 0] = (kp[:, 0] - c[0]) * s + CROP / 2
        kp[:, 1] = (kp[:, 1] - c[1]) * s + CROP / 2
        kp[kp[:, 2] == 0, :2] = 0
        rows.append((a["id"], a["image_id"], kp, c, side, a["area"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-kpts", type=int, default=None)
    ap.add_argument("--images-dir", type=Path, default=None, help="thư mục ảnh {split}2017 có sẵn")
    ap.add_argument("--limit-images", type=int, default=0, help="chỉ xử lý N ảnh (chạy thử)")
    args = ap.parse_args()
    images_dir = args.images_dir
    if images_dir is None and args.split == "val" and DEFAULT_VAL_IMAGES.exists():
        images_dir = DEFAULT_VAL_IMAGES

    d = json.load(open(ANN_DIR / f"person_keypoints_{args.split}2017.json"))
    # train: lọc người đủ rõ để học; val: giữ mọi người có >=1 khớp (đánh giá chuẩn COCO)
    min_k = args.min_kpts if args.min_kpts is not None else (5 if args.split == "train" else 1)
    min_area = 48 * 48 if args.split == "train" else 0
    anns = [a for a in d["annotations"]
            if not a["iscrowd"] and a["num_keypoints"] >= min_k and a["bbox"][2] * a["bbox"][3] >= min_area]
    by_img = {}
    for a in anns:
        by_img.setdefault(a["image_id"], []).append(a)
    if args.limit_images:
        by_img = dict(list(by_img.items())[:args.limit_images])
    imgs = {i["id"]: i for i in d["images"]}
    out_dir = CROPS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.split}: {sum(map(len, by_img.values()))} người / {len(by_img)} ảnh, "
          f"nguồn ảnh: {images_dir or 'images.cocodataset.org'}", flush=True)

    rows, done = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_image, args.split, imgs[i], a, out_dir, images_dir) for i, a in by_img.items()]
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(by_img)} ảnh, {len(rows)} crop", flush=True)

    rows.sort(key=lambda r: r[0])
    np.savez(CROPS_DIR / f"{args.split}_index.npz",
             ann_id=np.array([r[0] for r in rows], np.int64),
             image_id=np.array([r[1] for r in rows], np.int64),
             keypoints=np.stack([r[2] for r in rows]),
             center=np.stack([r[3] for r in rows]),
             side=np.array([r[4] for r in rows], np.float32),
             area=np.array([r[5] for r in rows], np.float32))
    print(f"Xong: {len(rows)} crop -> {CROPS_DIR / f'{args.split}_index.npz'}")


if __name__ == "__main__":
    main()
