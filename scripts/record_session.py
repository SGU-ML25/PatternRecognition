"""[P0] Thu tập dữ liệu nhỏ từ webcam: chuỗi khớp + nhãn động tác -> data/sessions/*.npz

Chạy (máy local có webcam + checkpoint đã train):

  SKYPOSE_CKPT=checkpoints/resnet18/best.pt \\
  .venv/bin/python scripts/record_session.py --person ten_ban --distance 1m --rounds 4

Script chạy ĐÚNG pipeline của game: cắt giữa khung 3:4, cạnh dài 480px, JPEG q=0.8, KHÔNG lật gương
(web/pose.js), rồi PoseSession (tracker + flip-test + One-Euro). Lưu cả khớp thô (trước One-Euro) để so bộ lọc.

Hai chế độ gán nhãn:
  guided (mặc định)  Màn hình nhắc từng động tác theo lịch; 2s "chuẩn bị" trước mỗi đoạn được gán nhãn -1
                     (bỏ qua) để không dính đoạn chuyển tiếp. Nhãn chính xác theo thời gian, không phải bấm phím.
  manual             Tự bấm phím để gán nhãn theo thời gian thực: 1 idle  2 left  3 right  4 flap  5 dive  0 bỏ qua

Cách quay đủ cho mục B (mỗi session ~3 phút với --rounds 4):
  - 3 session ở 1m (--distance 1m)     -> chọn ngưỡng + kiểm leave-one-session-out
  - 2 session ở 2m (--distance 2m)     -> thí nghiệm train 1m / test 2m (đặc trưng chuẩn hoá vs thô)
  - nên đổi áo / nền / ánh sáng giữa các session để các lần quay thật sự độc lập
Phím: q hoặc ESC = dừng và lưu.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.common import (CLASSES, DIVE, FLAP, IDLE, IGNORE, LEFT, PROMPTS, RIGHT,  # noqa: E402
                             save_session)

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
SEND_LONG, JPEG_Q = 480, 80  # như web/pose.js


# ---------------------------------------------------------------- lịch động tác
def build_protocol(rounds=4, seed=0, shuffle=True, ready=2.0, durations=None):
    """-> list[(label, duration_s)] : mỗi đoạn có nhãn đứng sau một đoạn IGNORE `ready` giây.
    Mỗi vòng: idle, {left,right,flap,dive} (xáo trộn), idle."""
    d = {IDLE: 6.0, LEFT: 5.0, RIGHT: 5.0, FLAP: 6.0, DIVE: 5.0}
    d.update(durations or {})
    rng = np.random.default_rng(seed)
    seq = []
    for _ in range(rounds):
        mid = [LEFT, RIGHT, FLAP, DIVE]
        if shuffle:
            rng.shuffle(mid)
        for lab in [IDLE] + mid + [IDLE]:
            seq += [(IGNORE, ready), (lab, d[lab])]
    return seq


def timeline(protocol):
    """-> list[(t_start, t_end, label)]"""
    out, t = [], 0.0
    for lab, dur in protocol:
        out.append((t, t + dur, lab)); t += dur
    return out


def state_at(tl, t):
    """(label, index đoạn, giây còn lại, nhãn đoạn kế tiếp) tại thời điểm t; None nếu hết lịch."""
    for i, (a, b, lab) in enumerate(tl):
        if a <= t < b:
            nxt = next((l for _, _, l in tl[i:] if l != IGNORE), None)
            return lab, i, b - t, nxt
    return None


# ---------------------------------------------------------------- tiền xử lý khung giống web client
def center_crop_resize(frame, aspect=3 / 4, long_side=SEND_LONG):
    """Cắt giữa video theo `aspect` (rộng/cao) rồi thu nhỏ cạnh dài = long_side — như PoseClient._setupCrop."""
    vh, vw = frame.shape[:2]
    w, h = vw, vh
    if vw / vh > aspect:
        w = round(vh * aspect)
    else:
        h = round(vw / aspect)
    x0, y0 = int((vw - w) / 2), int((vh - h) / 2)
    k = long_side / max(w, h)
    return cv2.resize(frame[y0:y0 + h, x0:x0 + w], (round(w * k), round(h * k)), interpolation=cv2.INTER_LINEAR)


def jpeg_roundtrip(img, quality=JPEG_Q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


# ---------------------------------------------------------------- vẽ giao diện
def draw_ui(img, kp, state, mode, n, fps, manual_label, person_seen):
    h, w = img.shape[:2]
    vis = cv2.flip(img, 1)  # chỉ để HIỂN THỊ cho người quay dễ làm theo như gương; dữ liệu lưu không lật
    if kp is not None:
        P = [(int(w - 1 - x), int(y)) for x, y, _ in kp]
        for a, b in SKELETON:
            if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                cv2.line(vis, P[a], P[b], (235, 220, 80), 2, cv2.LINE_AA)
        for j in range(5, 13):
            if kp[j, 2] > 0.3:
                cv2.circle(vis, P[j], 4, (40, 200, 255), -1, cv2.LINE_AA)
    bar = np.zeros((86, w, 3), np.uint8)
    if mode == "guided" and state:
        lab, _, left, nxt = state
        if lab == IGNORE:
            txt, col = f"CHUAN BI ({left:.0f}s): " + (PROMPTS[nxt] if nxt is not None else ""), (0, 200, 255)
        else:
            txt, col = f"{CLASSES[lab].upper()} ({left:.0f}s): {PROMPTS[lab]}", (80, 255, 120)
    elif mode == "manual":
        txt, col = (("GAN NHAN: " + CLASSES[manual_label].upper()) if manual_label >= 0 else "BO QUA (bam 1-5 de gan nhan)"), (80, 255, 120)
    else:
        txt, col = "XONG", (255, 255, 255)
    scale = min(0.5, 0.95 * w / max(cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0], 1))
    cv2.putText(bar, txt, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    cv2.putText(bar, f"khung {n}  {fps:.0f} FPS  " + ("" if person_seen else "KHONG THAY NGUOI - lui ra de thay vai+tay"),
                (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255) if person_seen else (0, 0, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


# ---------------------------------------------------------------- vòng ghi
def record(session, get_raw, cap, protocol, *, mode="guided", aspect=3 / 4, jpeg=True, show=True,
           clock=time.perf_counter, max_frames=None):
    """session.process(img, t) -> (kp_filtered(17,3), box); get_raw() -> khớp thô của lần process gần nhất."""
    tl = timeline(protocol)
    total = tl[-1][1] if tl else float("inf")
    T, KP, RAW, LAB, MS = [], [], [], [], []
    manual_label, t0, fps, wh = IGNORE, None, 0.0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Không đọc được khung từ camera"); break
            img = center_crop_resize(frame, aspect)
            wh = (img.shape[1], img.shape[0])
            if jpeg:
                img = jpeg_roundtrip(img)
            now = clock()
            t0 = now if t0 is None else t0
            t = now - t0
            if mode == "guided" and t >= total:
                break
            st = state_at(tl, t) if mode == "guided" else None
            lab = st[0] if st else manual_label
            tp = clock()
            kp, _ = session.process(img, now)
            ms = (clock() - tp) * 1e3
            raw = get_raw()
            T.append(t); KP.append(kp.copy()); RAW.append(raw.copy()); LAB.append(lab); MS.append(ms)
            if len(T) > 1:
                fps = 0.9 * fps + 0.1 / max(T[-1] - T[-2], 1e-3) if fps else 1 / max(T[-1] - T[-2], 1e-3)
            if max_frames and len(T) >= max_frames:
                break
            if show:
                seen = kp[5, 2] > 0.3 and kp[6, 2] > 0.3
                cv2.imshow("record_session (q = dung & luu)", draw_ui(img, kp, st, mode, len(T), fps, manual_label, seen))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if mode == "manual" and chr(key) in "012345" and key != 255:
                    manual_label = IGNORE if key == ord("0") else int(chr(key)) - 1
    finally:
        if show:
            cv2.destroyAllWindows()
    return (np.array(T), np.array(KP), np.array(RAW), np.array(LAB), np.array(MS), wh)


def summarize(t, label):
    print(f"\nĐã ghi {len(t)} khung, {t[-1]:.0f}s, ~{len(t) / max(t[-1], 1e-3):.1f} FPS")
    for c, name in enumerate(CLASSES):
        n = int((label == c).sum())
        print(f"  {name:6s} {n:5d} khung ({n / len(t) * 100:.0f}%)")
    print(f"  bỏ qua {int((label == IGNORE).sum()):5d} khung")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("SKYPOSE_CKPT", str(ROOT / "checkpoints" / "resnet18" / "best.pt")))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--person", default="p1", help="tên/mã người quay (ghi vào tên file + meta)")
    ap.add_argument("--distance", default="1m", help="khoảng cách tới camera: 1m / 2m ...")
    ap.add_argument("--notes", default="", help="ghi chú: áo, nền, ánh sáng...")
    ap.add_argument("--mode", choices=["guided", "manual"], default="guided")
    ap.add_argument("--rounds", type=int, default=4, help="số vòng (guided); mỗi vòng ~43s")
    ap.add_argument("--seed", type=int, default=0, help="seed xáo thứ tự động tác")
    ap.add_argument("--landscape", action="store_true", help="khung 4:3 như ?cam=ngang của game")
    ap.add_argument("--no-jpeg", action="store_true", help="bỏ bước nén JPEG q=0.8 (mặc định bật cho giống game)")
    ap.add_argument("--out", default=str(ROOT / "data" / "sessions"))
    args = ap.parse_args()

    import torch  # noqa: F401  (import muộn để --help chạy được khi chưa cài torch)
    from server.pose_engine import PoseEngine, PoseSession
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Không thấy checkpoint {args.ckpt}. Tải best.pt từ Kaggle run (xem README) hoặc đặt SKYPOSE_CKPT.")
    engine = PoseEngine(args.ckpt)
    holder = {}
    orig = engine.predict_crop

    def wrapped(*a, **k):
        holder["raw"] = orig(*a, **k)
        return holder["raw"]
    engine.predict_crop = wrapped
    session = PoseSession(engine)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)  # như web/pose.js
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {args.camera}")
    cam_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    aspect = 4 / 3 if args.landscape else 3 / 4
    proto = build_protocol(args.rounds, args.seed) if args.mode == "guided" else []
    if proto:
        print(f"Lịch quay: {timeline(proto)[-1][1]:.0f}s. Đứng cách camera {args.distance}, thấy rõ 2 vai + 2 tay dang ngang.")
        print("Bắt đầu sau 3 giây... (q/ESC để dừng và lưu)"); time.sleep(3)
    t, kp, raw, lab, ms, wh = record(session, lambda: holder["raw"], cap, proto, mode=args.mode, aspect=aspect,
                                 jpeg=not args.no_jpeg)
    cap.release()
    if len(t) < 10:
        raise SystemExit("Quá ít khung, không lưu.")
    meta = {"person": args.person, "distance": args.distance, "notes": args.notes, "mode": args.mode,
            "frame_wh": list(wh), "camera_wh": list(cam_wh), "aspect": "landscape" if args.landscape else "portrait",
            "ckpt": str(args.ckpt), "jpeg_q": None if args.no_jpeg else JPEG_Q, "conf_thr": session.conf_thr,
            "labels": CLASSES, "protocol": proto, "seed": args.seed, "date": datetime.now().isoformat(timespec="seconds")}
    path = Path(args.out) / f"{args.person}_{args.distance}_{datetime.now():%Y%m%d-%H%M%S}.npz"
    save_session(path, t, kp, raw, lab, ms, meta)
    summarize(t, lab)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
