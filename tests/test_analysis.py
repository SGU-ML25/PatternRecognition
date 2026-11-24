"""Chạy:  python tests/test_analysis.py   (không cần pytest, không cần torch/webcam)"""
import ast
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.common import load_session, save_session  # noqa: E402
from analysis.features import ControlParams, controls_replay, flap_velocity, frame_features  # noqa: E402
from analysis.filters import OneEuro  # noqa: E402
from analysis.synth import make_session  # noqa: E402
from server.controls import FlightControls  # noqa: E402


def test_replay_matches_flightcontrols():
    """Bản vector hoá phải cho kết quả TRÙNG controls.py (tới sai số làm tròn 4 chữ số của controls.py)."""
    for dist, seed in [(1.0, 0), (2.0, 1)]:
        s = make_session(dist, seed)
        kp, t = s["kp"].copy(), s["t"]
        # cố tình làm hỏng vài khung để chạy cả nhánh fallback khuỷu / mất vai
        rng = np.random.default_rng(5)
        kp[rng.random(len(t)) < 0.05, 9, 2] = 0.1     # mất cổ tay trái -> dùng khuỷu
        kp[rng.random(len(t)) < 0.02, 5, 2] = 0.1     # mất vai trái
        kp[rng.random(len(t)) < 0.02, 7, 2] = 0.1
        kp[rng.random(len(t)) < 0.02, 9, 2] = 0.1     # có khung mất cả cổ tay + khuỷu
        fc = FlightControls()
        ref = [fc(kp[i], t[i]) for i in range(len(t))]
        got = controls_replay(frame_features(kp), t, ControlParams())
        for key_ref, key_got in [("bank", "bank"), ("flap", "flap"), ("dive", "dive"),
                                 ("spread", "spread"), ("arms_up", "arms_up")]:
            a = np.array([r[key_ref] for r in ref])
            b = got[key_got]
            err = np.abs(a - b).max()
            assert err < 2e-3, f"{key_ref} lệch {err} (dist={dist})"
    print("OK  controls_replay == FlightControls")


def test_scale_invariance():
    """Nhân toạ độ với k (đứng gần/xa hơn) không đổi đặc trưng chuẩn hoá."""
    s = make_session(1.0, 3)
    F1 = frame_features(s["kp"])
    kp2 = s["kp"].copy(); kp2[:, :, :2] = kp2[:, :, :2] * 0.5 + 40
    F2 = frame_features(kp2)
    for k in ["roll", "spread", "below", "elev", "wing_l", "wing_r"]:
        assert np.allclose(F1[k], F2[k], equal_nan=True, atol=1e-6), k
    print("OK  đặc trưng bất biến tỉ lệ/tịnh tiến")


def test_oneeuro_same_as_server():
    """analysis.filters.OneEuro phải y hệt lớp trong server/pose_engine.py."""
    src = (ROOT / "server" / "pose_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OneEuro")
    ns = {"np": np}
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    rng = np.random.default_rng(0)
    a, b = ns["OneEuro"](), OneEuro()
    for i in range(200):
        x = rng.normal(size=(17, 2)); t = i * 0.033
        assert np.allclose(a(x, t), b(x, t))
    print("OK  OneEuro trùng server")


def test_session_roundtrip(tmp="/tmp/_sess_test.npz"):
    s = make_session(1.0, 0)
    save_session(tmp, s["t"], s["kp"], s["kp_raw"], s["label"], s["proc_ms"], s["meta"])
    r = load_session(tmp)
    assert np.allclose(r["kp"], s["kp"], atol=1e-3) and r["distance"] == 1.0 and (r["label"] == s["label"]).all()
    print("OK  save/load session")


def test_record_session_logic():
    """Kiểm tra logic thu dữ liệu bằng camera + model giả (không cần webcam/torch)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import record_session as rs
    from analysis.common import FLAP, IGNORE

    proto = rs.build_protocol(rounds=2, seed=1)
    tl = rs.timeline(proto)
    assert tl[0][2] == IGNORE and sum(1 for x in tl if x[2] == FLAP) == 2
    assert rs.state_at(tl, 0.5)[0] == IGNORE and rs.state_at(tl, 2.5)[0] == 0 and rs.state_at(tl, 1e6) is None

    # khung 1280x960 (4:3) -> cắt dọc 3:4 -> 360x480 như game
    fr = (np.random.default_rng(0).random((960, 1280, 3)) * 255).astype(np.uint8)
    out = rs.center_crop_resize(fr)
    assert out.shape == (480, 360, 3), out.shape
    assert rs.center_crop_resize(fr, 4 / 3).shape == (360, 480, 3)
    assert rs.jpeg_roundtrip(out).shape == out.shape

    src = make_session(1.0, 7, rounds=1)

    class FakeCap:
        def read(self): return True, fr
    class FakeSession:
        i = 0
        def process(self, img, t):
            kp = src["kp"][self.i]; self.raw = src["kp_raw"][self.i]; self.i += 1
            return kp, None
    sess = FakeSession(); tick = {"t": 0.0}
    def clock():
        tick["t"] += 1 / 60
        return tick["t"]
    t_, kp, raw, lab, ms, wh = rs.record(sess, lambda: sess.raw, FakeCap(), proto[:6], show=False, clock=clock)
    assert wh == (360, 480) and len(t_) == len(kp) == len(raw) == len(lab) > 100
    assert set(np.unique(lab)) <= {-1, 0, 1, 2, 3, 4}
    # nhãn đúng lịch: ~5.7s -> khung đầu là IGNORE
    assert lab[0] == IGNORE
    path = save_session("/tmp/_rec_test.npz", t_, kp, raw, lab, ms, {"distance": "1m", "frame_wh": list(wh)})
    assert load_session(path)["distance"] == 1.0
    print("OK  record_session: lịch, crop 3:4, JPEG, vòng ghi, lưu/đọc")


if __name__ == "__main__":
    test_record_session_logic()
    test_replay_matches_flightcontrols()
    test_scale_invariance()
    test_oneeuro_same_as_server()
    test_session_roundtrip()
    print("ALL PASSED")
