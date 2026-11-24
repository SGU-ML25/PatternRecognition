"""Các bộ lọc tín hiệu khớp để so sánh: không lọc / EMA / One-Euro / Kalman (vận tốc không đổi).

OneEuro ở đây giống hệt lớp trong server/pose_engine.py (copy lại vì file đó import torch).
Lọc chạy trên toạ độ chuẩn hoá /max(h,w) như PoseSession.process.
"""
import numpy as np


class OneEuro:
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


class EMA:
    """y_t = a*x_t + (1-a)*y_{t-1}  (a nhỏ = mượt hơn, trễ hơn)."""

    def __init__(self, alpha=0.5):
        self.a, self.y = alpha, None

    def __call__(self, x, t):
        self.y = x.copy() if self.y is None else self.y + self.a * (x - self.y)
        return self.y.copy()


class Kalman:
    """Kalman vận tốc không đổi cho từng toạ độ: trạng thái (x, v), nhiễu gia tốc q, nhiễu đo r."""

    def __init__(self, q=50.0, r=1e-5):
        self.q, self.r = q, r
        self.x = self.v = self.P = self.t = None

    def __call__(self, z, t):
        if self.x is None:
            self.x, self.v, self.t = z.copy(), np.zeros_like(z), t
            self.P = np.stack([np.full(z.shape, self.r), np.zeros(z.shape), np.full(z.shape, 1.0)])  # Pxx,Pxv,Pvv
            return z
        dt = max(t - self.t, 1e-3)
        Pxx, Pxv, Pvv = self.P
        x_p, v_p = self.x + self.v * dt, self.v
        q = self.q
        Pxx_p = Pxx + 2 * dt * Pxv + dt * dt * Pvv + q * dt ** 4 / 4
        Pxv_p = Pxv + dt * Pvv + q * dt ** 3 / 2
        Pvv_p = Pvv + q * dt ** 2
        S = Pxx_p + self.r
        Kx, Kv = Pxx_p / S, Pxv_p / S
        res = z - x_p
        self.x, self.v = x_p + Kx * res, v_p + Kv * res
        self.P = np.stack([(1 - Kx) * Pxx_p, (1 - Kx) * Pxv_p, Pvv_p - Kv * Pxv_p])
        self.t = t
        return self.x.copy()


def make_filter(kind, **kw):
    return {"none": lambda **_: None, "ema": EMA, "oneeuro": OneEuro, "kalman": Kalman}[kind](**kw)


def apply_filter(kp_raw, t, wh, kind, **kw):
    """kp_raw (T,17,3) pixel -> khớp đã lọc (cùng định dạng). Lọc trong toạ độ chuẩn hoá [0,1]."""
    if kind == "none":
        return kp_raw.copy()
    f = make_filter(kind, **kw)
    scale = float(max(wh))
    out = kp_raw.copy()
    for i in range(len(t)):
        out[i, :, :2] = f(kp_raw[i, :, :2] / scale, t[i]) * scale
    return out
