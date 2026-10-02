"""Hàm dùng chung cho train / evaluate / server: định nghĩa khớp, heatmap, giải mã."""
import cv2
import numpy as np

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
NUM_KPTS = 17
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = list(range(NUM_KPTS))
for a, b in FLIP_PAIRS:
    FLIP_INDEX[a], FLIP_INDEX[b] = b, a
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
# Độ lệch chuẩn OKS của COCO cho từng khớp
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                       1.07, 1.07, .87, .87, .89, .89]) / 10.0

INPUT_SIZE = 256
HEATMAP_SIZE = 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def square_affine(center, side, out_size, rot_deg=0.0, flip=False):
    """Ma trận 2x3 đưa hình vuông (center, side, xoay rot) của ảnh nguồn về ảnh out_size x out_size."""
    s = out_size / side
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rot_deg, s)
    M[0, 2] += out_size / 2 - center[0]
    M[1, 2] += out_size / 2 - center[1]
    if flip:
        M = np.array([[-1, 0, out_size - 1], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
    return M.astype(np.float32)


def invert_affine(M):
    return cv2.invertAffineTransform(M)


def apply_affine(pts, M):
    pts = np.asarray(pts, np.float32)
    return pts @ M[:, :2].T + M[:, 2]


def to_tensor_input(img_bgr):
    """BGR uint8 HxWx3 -> float32 3xHxW đã chuẩn hoá ImageNet (dùng khi suy luận từng ảnh)."""
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1).copy()


def to_uint8_chw(img_bgr):
    """BGR uint8 HxWx3 -> RGB uint8 3xHxW. Dataset trả dạng này (nhẹ hơn 4 lần so với float32),
    việc chuẩn hoá làm trên GPU bằng normalize_batch()."""
    return np.ascontiguousarray(img_bgr[:, :, ::-1].transpose(2, 0, 1))


def normalize_batch(x):
    """Tensor uint8 (N,3,H,W) RGB trên GPU -> float32 chuẩn hoá ImageNet, channels_last.
    Cho kết quả giống hệt to_tensor_input()."""
    import torch
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    x = (x.float() / 255.0 - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


def make_heatmaps(kp_in, vis, sigma=2.0):
    """kp_in: (17,2) toạ độ trong ảnh input 256 -> heatmap (17,64,64), weight (17,)."""
    stride = INPUT_SIZE / HEATMAP_SIZE
    hm = np.zeros((NUM_KPTS, HEATMAP_SIZE, HEATMAP_SIZE), np.float32)
    w = (vis > 0).astype(np.float32)
    xs = np.arange(HEATMAP_SIZE, dtype=np.float32)
    for j in range(NUM_KPTS):
        if w[j] == 0:
            continue
        mx, my = kp_in[j] / stride
        if not (-3 * sigma <= mx < HEATMAP_SIZE + 3 * sigma and -3 * sigma <= my < HEATMAP_SIZE + 3 * sigma):
            w[j] = 0
            continue
        gx = np.exp(-((xs - mx) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((xs - my) ** 2) / (2 * sigma ** 2))
        hm[j] = gy[:, None] * gx[None, :]
    return hm, w


def decode_heatmaps(hm):
    """hm: (N,17,H,W) numpy -> coords (N,17,2) trong hệ heatmap, conf (N,17).

    argmax + dịch 1/4 pixel về phía giá trị lân cận lớn hơn (SimpleBaseline)."""
    N, K, H, W = hm.shape
    flat = hm.reshape(N, K, -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1)
    xs = (idx % W).astype(np.float32)
    ys = (idx // W).astype(np.float32)
    for n in range(N):
        for k in range(K):
            x, y = int(xs[n, k]), int(ys[n, k])
            h = hm[n, k]
            if 0 < x < W - 1:
                xs[n, k] += 0.25 * np.sign(h[y, x + 1] - h[y, x - 1])
            if 0 < y < H - 1:
                ys[n, k] += 0.25 * np.sign(h[y + 1, x] - h[y - 1, x])
    return np.stack([xs, ys], -1), conf


def heatmap_to_input(coords):
    """Toạ độ heatmap -> toạ độ ảnh input (cùng quy ước với make_heatmaps)."""
    return coords * (INPUT_SIZE / HEATMAP_SIZE)
