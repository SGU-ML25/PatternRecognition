"""Hằng số + đọc/ghi session + cửa sổ trượt dùng chung cho phần Feature engineering (mục B).

Một *session* = một lần quay webcam, lưu `.npz` (xem scripts/record_session.py):
    t        (T,)      giây kể từ lúc bắt đầu
    kp       (T,17,3)  khớp SAU One-Euro (đúng thứ server đưa vào controls.py): x, y (pixel), conf
    kp_raw   (T,17,3)  khớp TRƯỚC lọc (để so sánh bộ lọc)
    label    (T,)      nhãn động tác, -1 = bỏ qua (đoạn chuyển tiếp)
    proc_ms  (T,)      thời gian chạy model + lọc cho mỗi khung
    meta     JSON      distance, person, frame_wh, ...
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Nhãn động tác. IDLE = "đứng yên ở tư thế lượn": dang ngang 2 tay, giữ thẳng, KHÔNG vỗ.
# (Tay buông thõng sát người = dive theo thiết kế của controls.py, nên không dùng làm IDLE.)
CLASSES = ["idle", "left", "right", "flap", "dive"]
IDLE, LEFT, RIGHT, FLAP, DIVE = range(5)
IGNORE = -1

PROMPTS = {  # chỉ ASCII vì cv2.putText không vẽ được dấu tiếng Việt
    IDLE: "DUNG YEN: dang ngang 2 tay, giu thang",
    LEFT: "NGHIENG TRAI: ha tay TRAI thap, tay phai cao",
    RIGHT: "NGHIENG PHAI: ha tay PHAI thap, tay trai cao",
    FLAP: "VO CANH: dang tay roi vay manh XUONG nhieu lan",
    DIVE: "BO NHAO: khep 2 tay sat nguoi, ha thap",
}


def save_session(path, t, kp, kp_raw, label, proc_ms=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, np.float64)
    np.savez_compressed(
        path, t=t, kp=np.asarray(kp, np.float32), kp_raw=np.asarray(kp_raw, np.float32),
        label=np.asarray(label, np.int8),
        proc_ms=np.zeros(len(t), np.float32) if proc_ms is None else np.asarray(proc_ms, np.float32),
        meta=json.dumps(meta or {}, ensure_ascii=False))
    return path


def _parse_distance(s):
    m = re.search(r"[\d.]+", str(s or ""))
    return float(m.group()) if m else float("nan")


def load_session(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    t = z["t"].astype(np.float64)
    return {"name": Path(path).stem, "t": t - t[0], "kp": z["kp"].astype(np.float64),
            "kp_raw": z["kp_raw"].astype(np.float64), "label": z["label"].astype(int),
            "proc_ms": z["proc_ms"], "meta": meta, "distance": _parse_distance(meta.get("distance")),
            "wh": tuple(meta.get("frame_wh", (360, 480)))}


def load_dir(path, pattern="*.npz"):
    files = sorted(Path(path).glob(pattern))
    if not files:
        raise SystemExit(f"Không thấy file {pattern} trong {path}. Quay dữ liệu bằng scripts/record_session.py "
                         f"hoặc thêm --synthetic để chạy thử pipeline.")
    return [load_session(f) for f in files]


def get_sessions(args):
    """Dùng chung cho các CLI: --data DIR hoặc --synthetic."""
    if getattr(args, "synthetic", False):
        from .synth import make_dataset
        print("!! DỮ LIỆU GIẢ LẬP — chỉ để kiểm tra pipeline chạy được; KHÔNG đưa số liệu này vào báo cáo.")
        return make_dataset(seed=0)
    return load_dir(args.data)


def add_data_args(ap):
    ap.add_argument("--data", default=str(ROOT / "data" / "sessions"), help="thư mục chứa *.npz")
    ap.add_argument("--synthetic", action="store_true", help="dùng dữ liệu giả lập để chạy thử")
    ap.add_argument("--out", default=str(ROOT / "report" / "feature_eval"), help="thư mục kết quả")
    ap.add_argument("--win", type=float, default=0.5, help="độ dài cửa sổ trượt (giây)")
    return ap


# ---------------------------------------------------------------- cửa sổ trượt (nhân quả)
def causal_bounds(t, win):
    """lo[i] = chỉ số đầu tiên của cửa sổ (t_i - win, t_i]."""
    return np.searchsorted(t, t - win, side="right")


def _ts(x, t):
    return pd.Series(x, index=pd.to_timedelta(t, unit="s"))


def roll_agg(x, t, win, how, min_periods=3):
    """Thống kê cuộn theo thời gian (nhân quả, bỏ NaN) -> (T,) numpy."""
    r = _ts(x, t).rolling(f"{int(round(win * 1000))}ms", min_periods=min_periods)
    return getattr(r, how)().to_numpy()


def window_stats(B, names, t, win):
    """B (T,D) -> (T, D*5): mean, std, min, max, delta(=độ dốc/giây) của từng đặc trưng trên cửa sổ trượt."""
    T, D = B.shape
    lo = causal_bounds(t, win)
    dt = np.maximum(t - t[lo], 1e-3)
    cols, out_names = [], []
    for d in range(D):
        x = B[:, d]
        for how in ("mean", "std", "min", "max"):
            cols.append(roll_agg(x, t, win, how))
            out_names.append(f"{names[d]}:{how}")
        cols.append((x - x[lo]) / dt)
        out_names.append(f"{names[d]}:slope")
    return np.stack(cols, 1), out_names


def label_purity(label, t, win):
    """Cửa sổ 'sạch' = đủ độ dài, mọi khung cùng một nhãn và nhãn >= 0."""
    lab = label.astype(float)
    mn, mx = roll_agg(lab, t, win, "min", 1), roll_agg(lab, t, win, "max", 1)
    return (mn == mx) & (mn >= 0) & (t - t[0] >= win)
