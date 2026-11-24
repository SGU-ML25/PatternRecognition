"""Sinh dữ liệu session GIẢ LẬP để kiểm tra pipeline chạy được.

!! Số liệu trên dữ liệu giả KHÔNG có giá trị khoa học (hình học tay + nhiễu do mình tự đặt, nên công thức
   đương nhiên 'đúng'). Chỉ dùng để test code và xem biểu đồ có vẽ được hay không.

Mô hình: người đứng giữa khung 360x480, vai rộng S pixel (1m: ~70px, 2m: ~35px), mỗi tay dài 1.4S,
góc tay theta so với phương ngang (dương = giơ lên). Nhiễu đo Gaussian theo pixel (không phụ thuộc
khoảng cách -> ở 2m nhiễu tương đối lớn gấp đôi, giống thật).
"""
import numpy as np

from .common import DIVE, FLAP, IDLE, IGNORE, LEFT, RIGHT
from .features import LE, LH, LS, LW, RE, RH, RS, RW
from .filters import apply_filter

W, H = 360, 480
ARM, ELB = 1.4, 0.75 / 1.4  # chiều dài tay (đơn vị S), vị trí khuỷu trên tay


def _protocol(rng, rounds):
    seq = []
    for _ in range(rounds):
        mid = [LEFT, RIGHT, FLAP, DIVE]
        rng.shuffle(mid)
        for lab, dur in [(IDLE, 6)] + [(m, 5 if m != FLAP else 6) for m in mid] + [(IDLE, 4)]:
            seq += [(IGNORE, 2.0), (lab, float(dur))]
    return seq


def _target(lab, rng, state):
    """(theta_left, theta_right) mục tiêu, độ."""
    a = rng.uniform(22, 40)
    return {IDLE: (0, 0), LEFT: (-a, a), RIGHT: (a, -a), DIVE: (-80, -80), FLAP: (0, 0)}[lab]


def make_session(distance=1.0, seed=0, rounds=3, fps=30.0, noise_px=1.6, name=None, person="synth"):
    rng = np.random.default_rng(seed)
    S0 = 70.0 / distance
    seq = _protocol(rng, rounds)
    t, lab = [], []
    now, dt0 = 0.0, 1 / fps
    for L, dur in seq:
        for _ in range(int(dur * fps)):
            t.append(now + rng.normal(0, 0.003)); lab.append(L); now += dt0
    t = np.array(t); lab = np.array(lab)
    # tư thế mục tiêu theo từng khung (đoạn IGNORE lấy tư thế của đoạn sau)
    targets = np.zeros((len(t), 2)); is_flap = np.zeros(len(t), bool)
    i = 0
    seg_info = []
    for L, dur in seq:
        n = int(dur * fps); seg_info.append((i, i + n, L)); i += n
    for k, (a, b, L) in enumerate(seg_info):
        eff = L if L != IGNORE else seg_info[k + 1][2]
        tg = _target(eff, rng, None)
        targets[a:b] = tg
        is_flap[a:b] = (eff == FLAP) and L != IGNORE
    theta = np.zeros((len(t), 2)); cur = np.zeros(2); flap_phase = 0.0
    f_hz = 1.8
    for i in range(len(t)):
        dti = t[i] - t[i - 1] if i else dt0
        cur += (targets[i] - cur) * min(1.0, dti / 0.18)  # chuyển tư thế mượt (hằng số thời gian ~0.18s)
        th = cur.copy()
        if is_flap[i]:
            flap_phase += 2 * np.pi * f_hz * dti
            th += 32 * np.sin(flap_phase) * np.array([1, 1]) + np.array([4, 4])
        theta[i] = th
    T = len(t)
    sway = np.cumsum(rng.normal(0, 0.25, (T, 2)), 0)  # trôi nhẹ của thân
    sway = np.clip(sway, -12, 12)
    c = np.array([W / 2, 210.0 + 30 * (distance - 1)])
    kp = np.zeros((T, 17, 3)); kp[:, :, 2] = 0.85
    for j in range(17):  # các khớp không dùng: đặt quanh thân cho đủ 17
        kp[:, j, :2] = c + sway + np.array([0, 40 * (j - 8)]) * S0 / 70 * 0.2
    for side, (sh, el, wr, sgn, ti) in {"L": (LS, LE, LW, +1, 0), "R": (RS, RE, RW, -1, 1)}.items():
        shp = c + sway + np.array([sgn * S0 / 2, 0])
        th = np.radians(theta[:, ti])
        d = np.stack([sgn * np.cos(th), -np.sin(th)], 1) * ARM * S0
        kp[:, sh, :2] = shp
        kp[:, el, :2] = shp + d * ELB
        kp[:, wr, :2] = shp + d
    kp[:, LH, :2] = c + sway + np.array([S0 * 0.35, 1.3 * S0 * 1.4]); kp[:, RH, :2] = c + sway + np.array([-S0 * 0.35, 1.3 * S0 * 1.4])
    kp[:, 0, :2] = c + sway + np.array([0, -0.7 * S0])
    kp_raw = kp.copy()
    kp_raw[:, :, :2] += rng.normal(0, noise_px, kp_raw[:, :, :2].shape)
    drop = rng.random((T, 17)) < 0.015  # thỉnh thoảng mất khớp
    kp_raw[:, :, 2] = np.where(drop, 0.1, rng.uniform(0.65, 0.95, (T, 17)))
    kp_f = apply_filter(kp_raw, t, (W, H), "oneeuro")
    meta = {"distance": f"{distance:g}m", "person": person, "frame_wh": [W, H], "synthetic": True}
    return {"name": name or f"synth_{distance:g}m_s{seed}", "t": t - t[0], "kp": kp_f, "kp_raw": kp_raw,
            "label": lab.astype(int), "proc_ms": np.full(T, 8.0), "meta": meta, "distance": float(distance),
            "wh": (W, H)}


def make_dataset(seed=0):
    """3 session ở 1m + 2 session ở 2m."""
    ss = [make_session(1.0, seed + k, name=f"synth_1m_{k}") for k in range(3)]
    ss += [make_session(2.0, seed + 10 + k, name=f"synth_2m_{k}") for k in range(2)]
    return ss
