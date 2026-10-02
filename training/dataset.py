"""Dataset COCO-Keypoints đọc từ các crop 320px do prepare_data.py tạo ra."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import CROPS_DIR as CROPS
from pose_utils import (FLIP_INDEX, INPUT_SIZE, apply_affine, make_heatmaps, square_affine,
                        to_tensor_input)

CROP = 320
BASE_SIDE = CROP / 1.5 * 1.25   # bbox nằm gọn, chừa 25% lề như SimpleBaseline
UPPER = list(range(13))         # đầu, vai, tay, hông — giống khung hình webcam


class CocoPoseCrops(Dataset):
    def __init__(self, split, train):
        self.split, self.train = split, train
        idx = np.load(CROPS / f"{split}_index.npz")
        self.ann_id = idx["ann_id"]
        self.image_id = idx["image_id"]
        self.kps = idx["keypoints"]
        self.center = idx["center"]
        self.side = idx["side"]
        self.area = idx["area"]

    def __len__(self):
        return len(self.ann_id)

    def _augment_box(self, kp):
        c = np.array([CROP / 2, CROP / 2], np.float32)
        side = BASE_SIDE
        vis = kp[:, 2] > 0
        # Half-body: chỉ lấy nửa người trên -> mô phỏng webcam
        up = [j for j in UPPER if vis[j]]
        if np.random.rand() < 0.4 and len(up) >= 4:
            pts = kp[up, :2]
            lo, hi = pts.min(0), pts.max(0)
            c = (lo + hi) / 2
            side = max(float((hi - lo).max()) * 1.5, 48.0)
        side *= np.random.uniform(0.7, 1.35)
        c = c + np.random.uniform(-0.1, 0.1, 2) * side
        rot = np.random.uniform(-30, 30) if np.random.rand() < 0.6 else 0.0
        flip = np.random.rand() < 0.5
        return c, side, rot, flip

    def _color(self, img):
        img = img.astype(np.float32)
        img = img * np.random.uniform(0.6, 1.4) + np.random.uniform(-30, 30)  # contrast, brightness
        if np.random.rand() < 0.5:  # saturation
            gray = img.mean(2, keepdims=True)
            img = gray + (img - gray) * np.random.uniform(0.5, 1.5)
        if np.random.rand() < 0.3:  # random erasing: mô phỏng che khuất / tay ra khỏi khung
            h, w = np.random.randint(20, 90, 2)
            y, x = np.random.randint(0, INPUT_SIZE - h), np.random.randint(0, INPUT_SIZE - w)
            img[y:y + h, x:x + w] = np.random.uniform(0, 255, 3)
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        img = cv2.imread(str(CROPS / self.split / f"{self.ann_id[i]}.jpg"))
        kp = self.kps[i].copy()
        if self.train:
            c, side, rot, flip = self._augment_box(kp)
        else:
            c, side, rot, flip = np.array([CROP / 2, CROP / 2]), BASE_SIDE, 0.0, False
        M = square_affine(c, side, INPUT_SIZE, rot, flip)
        inp = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR)
        kp_in = apply_affine(kp[:, :2], M)
        vis = kp[:, 2].copy()
        if flip:
            kp_in, vis = kp_in[FLIP_INDEX], vis[FLIP_INDEX]
        if self.train:
            inp = self._color(inp)
        hm, w = make_heatmaps(kp_in, vis)
        return (torch.from_numpy(to_tensor_input(inp)), torch.from_numpy(hm), torch.from_numpy(w),
                torch.from_numpy(M), i)
