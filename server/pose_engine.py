"""Suy luận Model 1 cho webcam: tracker vùng cắt + flip-test + lọc One-Euro."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from model import PoseNet  # noqa: E402
from pose_utils import (FLIP_INDEX, INPUT_SIZE, KEYPOINTS, decode_heatmaps,  # noqa: E402
                        heatmap_to_input, square_affine, to_tensor_input)

TRACK_KPTS = list(range(13))  # đầu, tay, hông: dùng để đặt vùng cắt khung sau


class OneEuro:
    """Bộ lọc One-Euro (Casiez 2012): ít rung khi đứng yên, ít trễ khi chuyển động nhanh."""

    def __init__(self, min_cutoff=1.0, beta=3.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x.copy()


class PoseEngine:
    def __init__(self, ckpt, device=None, flip=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.backbone = ck["backbone"]
        self.info = {"checkpoint": str(ckpt), "backbone": ck["backbone"], "epoch": ck.get("epoch"),
                     "coco_val": ck.get("coco"), "input_size": INPUT_SIZE, "keypoints": KEYPOINTS,
                     "device": self.device, "flip_test": flip}
        self.model = PoseNet(ck["backbone"], pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.model.half().to(memory_format=torch.channels_last)
        self.flip = flip
        self.predict_crop(np.zeros((480, 640, 3), np.uint8), (320, 240), 640)  # warm-up

    @torch.no_grad()
    def predict_crop(self, img, center, side):
        """Chạy model trên vùng vuông (center, side) -> (17,3): x, y (pixel ảnh gốc), conf."""
        M = square_affine(np.asarray(center, np.float32), side, INPUT_SIZE)
        inp = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        x = torch.from_numpy(to_tensor_input(inp))[None].to(self.device)
        if self.flip:
            x = torch.cat([x, torch.flip(x, [3])])
        x = x.half() if self.half else x
        hm = self.model(x.contiguous(memory_format=torch.channels_last)).float()
        if self.flip:
            hf = torch.flip(hm[1:], [3])[:, FLIP_INDEX]
            hf[:, :, :, 1:] = hf[:, :, :, :-1].clone()
            hm = (hm[:1] + hf) / 2
        coords, conf = decode_heatmaps(hm.cpu().numpy())
        xy_in = heatmap_to_input(coords[0])
        Minv = cv2.invertAffineTransform(M)
        xy = xy_in @ Minv[:, :2].T + Minv[:, 2]
        return np.concatenate([xy, conf[0][:, None]], 1)


class PoseSession:
    """Trạng thái theo từng người chơi (mỗi kết nối WebSocket): vùng cắt, bộ lọc."""

    def __init__(self, engine, conf_thr=0.3):
        self.engine, self.conf_thr = engine, conf_thr
        self.box = None  # (cx, cy, side)
        self.filters = [OneEuro() for _ in range(17)]

    def _full_box(self, h, w):
        return (w / 2, h / 2, float(max(h, w)))

    def process(self, img, t=None):
        t = time.perf_counter() if t is None else t
        h, w = img.shape[:2]
        box = self.box or self._full_box(h, w)
        kp = self.engine.predict_crop(img, box[:2], box[2])
        good = [j for j in TRACK_KPTS if kp[j, 2] > self.conf_thr]
        if len(good) >= 4 and kp[[5, 6], 2].min() > self.conf_thr:
            lo, hi = kp[good, :2].min(0), kp[good, :2].max(0)
            side = float(np.clip((hi - lo).max() * 1.6, 0.35 * max(h, w), 1.6 * max(h, w)))
            nb = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, side)
            # làm mượt vùng cắt để tránh giật
            self.box = nb if self.box is None else tuple(0.5 * np.array(self.box) + 0.5 * np.array(nb))
        else:
            self.box = None  # mất dấu -> khung sau dùng cả khung hình
        # lọc trong toạ độ chuẩn hoá [0,1] để tham số One-Euro không phụ thuộc độ phân giải
        out = kp.copy()
        scale = float(max(h, w))
        for j in range(17):
            out[j, :2] = self.filters[j](kp[j, :2] / scale, t) * scale
        return out, box
