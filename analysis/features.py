"""Feature engineering: 17 khớp -> đặc trưng điều khiển (bản vector hoá của server/controls.py).

Mọi khoảng cách chuẩn hoá theo độ rộng vai S = ||LS - RS||  =>  bất biến với khoảng cách tới camera.
`controls_replay` tái hiện CHÍNH XÁC FlightControls (có test trong tests/test_analysis.py) nhưng cho
phép đổi ngưỡng — cần cho việc chọn ngưỡng bằng lưới/ROC thay vì "đoán".
"""
from dataclasses import dataclass

import numpy as np

from .common import window_stats

NOSE, LS, RS, LE, RE, LW, RW, LH, RH = 0, 5, 6, 7, 8, 9, 10, 11, 12
UPPER = [LS, RS, LE, RE, LW, RW, LH, RH]
UPPER_NAMES = ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH"]
ENGINEERED = ["roll", "spread", "wing_l", "wing_r", "below", "elev", "v"]
VARIANTS = ["raw_px", "center_px", "center_scale", "engineered"]


def frame_features(kp, conf_thr=0.3):
    """kp (T,17,3) -> dict các mảng (T,). NaN ở khung không đủ khớp (giống controls.py trả về 0)."""
    xy, conf = kp[:, :, :2], kp[:, :, 2]
    sh_ok = (conf[:, LS] > conf_thr) & (conf[:, RS] > conf_thr)
    ls, rs = xy[:, LS], xy[:, RS]
    S = np.linalg.norm(ls - rs, axis=1) + 1e-6
    mid = (ls + rs) / 2

    def tip(w, e):  # đầu mút cánh: cổ tay nếu thấy, không thì khuỷu (Fallback khuỷu)
        w_ok, e_ok = conf[:, w] > conf_thr, conf[:, e] > conf_thr
        p = np.where(w_ok[:, None], xy[:, w], np.where(e_ok[:, None], xy[:, e], np.nan))
        return p, (~w_ok) & e_ok

    lt, lfb = tip(LW, LE)
    rt, rfb = tip(RW, RE)
    tips_ok = sh_ok & ~np.isnan(lt[:, 0]) & ~np.isnan(rt[:, 0])
    nan = np.full(len(kp), np.nan)

    def only(x):
        return np.where(tips_ok, x, nan)

    return {
        "sh_ok": sh_ok, "tips_ok": tips_ok, "S": S, "mid": mid,
        "roll": only(np.arctan2(lt[:, 1] - rt[:, 1], lt[:, 0] - rt[:, 0])),
        "spread": only(np.abs(lt[:, 0] - rt[:, 0]) / S),
        "wing_l": only(np.arctan2(ls[:, 1] - lt[:, 1], np.abs(lt[:, 0] - ls[:, 0]) + 1e-6)),
        "wing_r": only(np.arctan2(rs[:, 1] - rt[:, 1], np.abs(rt[:, 0] - rs[:, 0]) + 1e-6)),
        "below": only(((lt[:, 1] + rt[:, 1]) / 2 - mid[:, 1]) / S),
        "elev": only(((ls[:, 1] - lt[:, 1]) + (rs[:, 1] - rt[:, 1])) / 2 / S),
        "fallback": only((lfb | rfb).astype(float)),
    }


def flap_velocity(F, t, maxlen=8):
    """v = -d(elev)/dt (S/giây, >0 khi tay đi xuống) trên 8 mẫu gần nhất — như deque trong controls.py."""
    elev, sh_ok, tips_ok = F["elev"], F["sh_ok"], F["tips_ok"]
    v = np.full(len(t), np.nan)
    hist = []
    for i in range(len(t)):
        if not sh_ok[i]:
            hist.clear()
            continue
        if not tips_ok[i]:
            continue
        hist.append(i)
        hist = hist[-maxlen:]
        if len(hist) >= 3:
            j = hist[0]
            v[i] = -(elev[i] - elev[j]) / max(t[i] - t[j], 1e-3)
    return v


@dataclass
class ControlParams:
    """Các hằng số trong controls.py (mặc định = giá trị đang dùng)."""
    bank_full_deg: float = 40.0
    flap_v0: float = 1.0
    flap_scale: float = 4.0
    flap_spread: float = 1.8
    flap_decay: float = 0.85
    dive_spread0: float = 2.2
    dive_range: float = 1.2
    dive_below: float = 0.5


def flap_track(v, spread, sh_ok, tips_ok, v0=1.0, scale=4.0, spread_thr=1.8, decay=0.85):
    """Chuỗi `flap` (0..1) có giữ xung lực: flap = max(target, flap*decay)."""
    out = np.zeros(len(v))
    flap = 0.0
    for i in range(len(v)):
        if not sh_ok[i]:
            flap *= 0.8
            continue
        if not tips_ok[i]:
            continue
        tgt = 0.0
        if not np.isnan(v[i]) and spread[i] > spread_thr:
            tgt = min(1.0, max(0.0, (v[i] - v0) / scale))
        flap = max(tgt, flap * decay)
        out[i] = flap
    return out


def dive_score(spread, below, p=ControlParams()):
    s = np.clip((p.dive_spread0 - spread) / p.dive_range, 0, 1) * np.clip(below / p.dive_below, 0, 1)
    return np.nan_to_num(s)


def controls_replay(F, t, p=ControlParams(), v=None):
    """Tái hiện đầu ra của FlightControls trên cả chuỗi: bank, flap, dive, spread, arms_up (elev)."""
    v = flap_velocity(F, t) if v is None else v
    z = lambda x: np.nan_to_num(x)  # noqa: E731
    return {
        "bank": np.clip(z(F["roll"]) / np.radians(p.bank_full_deg), -1, 1),
        "flap": flap_track(v, F["spread"], F["sh_ok"], F["tips_ok"], p.flap_v0, p.flap_scale,
                           p.flap_spread, p.flap_decay),
        "dive": dive_score(F["spread"], F["below"], p),
        "spread": z(F["spread"]), "arms_up": z(F["elev"]),
    }


# ---------------------------------------------------------------- ma trận đặc trưng cho bộ phân loại học
def base_features(sess, variant, kp_key="kp", conf_thr=0.3):
    """(T,D) đặc trưng từng khung + tên. variant:
       raw_px       toạ độ pixel thô của 8 khớp thân trên
       center_px    trừ trung điểm hai vai (bất biến vị trí, KHÔNG bất biến tỉ lệ)
       center_scale chia thêm cho S (bất biến vị trí + tỉ lệ)
       engineered   7 đặc trưng điều khiển (roll, spread, wing_l/r, below, elev, v)"""
    kp, t = sess[kp_key], sess["t"]
    F = frame_features(kp, conf_thr)
    if variant == "engineered":
        v = flap_velocity(F, t)
        B = np.stack([F["roll"], F["spread"], F["wing_l"], F["wing_r"], F["below"], F["elev"], v], 1)
        return B, list(ENGINEERED)
    xy = kp[:, UPPER, :2]
    if variant in ("center_px", "center_scale"):
        xy = xy - F["mid"][:, None, :]
    if variant == "center_scale":
        xy = xy / F["S"][:, None, None]
    names = [f"{n}_{a}" for n in UPPER_NAMES for a in "xy"]
    return xy.reshape(len(kp), -1), names


def session_windows(sess, variant, win=0.5, kp_key="kp"):
    """Đặc trưng cửa sổ trượt (nhân quả) cho MỌI khung. Trả dict: X, names, y, pure, valid."""
    from .common import label_purity
    B, base = base_features(sess, variant, kp_key)
    X, names = window_stats(B, base, sess["t"], win)
    return {"X": X, "names": names, "base": base, "y": sess["label"],
            "pure": label_purity(sess["label"], sess["t"], win),
            "valid": sess["t"] - sess["t"][0] >= win}


def stack_windows(sessions, variant, win=0.5, stride=2):
    """Gộp các session: X, y, group (chỉ số session), distance — chỉ lấy cửa sổ sạch, lấy mỗi `stride` khung."""
    Xs, ys, gs, ds, names = [], [], [], [], None
    for k, s in enumerate(sessions):
        w = session_windows(s, variant, win)
        idx = np.where(w["pure"])[0][::stride]
        Xs.append(w["X"][idx]); ys.append(w["y"][idx])
        gs.append(np.full(len(idx), k)); ds.append(np.full(len(idx), s["distance"]))
        names = w["names"]
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ds), names
