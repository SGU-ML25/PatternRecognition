"""[P1] Phương án học máy cho cùng nhiệm vụ vs công thức tay.

  python -m analysis.compare_ml --data data/sessions

Đầu vào học máy: 7 đặc trưng điều khiển (engineered) -> cửa sổ trượt 0.5s: mean/std/min/max/độ dốc (35 chiều).
Mô hình: Logistic Regression / SVM(RBF) / Random Forest / MLP nhỏ. Công thức tay: ngưỡng chọn bằng lưới.
Mọi thứ kiểm bằng leave-one-session-out (không rò rỉ giữa các lần quay). So sánh: độ chính xác (macro-F1),
F1 của flap, báo nhầm khi đứng yên, độ trễ flap, thời gian suy luận từng mẫu, khả năng giải thích.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from server.controls import FlightControls  # noqa: E402

from .common import CLASSES, FLAP, IDLE, add_data_args, get_sessions  # noqa: E402
from .eval_controls import (fa_summary, flap_latencies, idle_false_alarm, lat_summary, loso, prepare,  # noqa: E402
                            report, tune)
from .features import session_windows, stack_windows  # noqa: E402
from .mlutil import MODELS, make_model  # noqa: E402

NICE = {"formula": "Công thức tay (ngưỡng chọn)", "logreg": "Logistic Regression", "svm": "SVM (RBF)",
        "rf": "Random Forest", "mlp": "MLP (32-16)"}
EXPLAIN = {"formula": "cao: mỗi lệnh = 1 công thức/ngưỡng có nghĩa vật lý", "logreg": "khá: hệ số tuyến tính",
           "svm": "thấp", "rf": "trung bình: độ quan trọng đặc trưng", "mlp": "thấp"}


def ml_loso(sessions, pres, win, kind):
    Xall, yall, gall, _, names = stack_windows(sessions, "engineered", win, stride=2)
    yt, yp, lat, fa, tr_t = [], [], [], [], []
    for k, (s, pre) in enumerate(zip(sessions, pres)):
        t0 = time.perf_counter()
        m = make_model(kind).fit(Xall[gall != k], yall[gall != k]); tr_t.append(time.perf_counter() - t0)
        w = session_windows(s, "engineered", win)
        pred = np.full(len(s["t"]), IDLE)
        pred[w["valid"]] = m.predict(w["X"][w["valid"]])  # dự đoán nhân quả cho từng khung
        yt.append(pre["y"][pre["pure"]]); yp.append(pred[pre["pure"]])
        lat += flap_latencies(pre, pred); fa.append(idle_false_alarm(pre, pred))
    # thời gian suy luận một mẫu (như khi chạy realtime)
    s0 = sessions[0]; w0 = session_windows(s0, "engineered", win)
    m = make_model(kind).fit(Xall, yall); xs = w0["X"][w0["valid"]][:200]
    t0 = time.perf_counter()
    for x in xs:
        m.predict(x[None])
    one_ms = (time.perf_counter() - t0) / len(xs) * 1e3
    return np.concatenate(yt), np.concatenate(yp), lat, fa, one_ms, float(np.mean(tr_t)), (m, names, Xall)


def formula_ms(sess):
    fc = FlightControls(); kp, t = sess["kp"], sess["t"]
    t0 = time.perf_counter()
    for i in range(len(t)):
        fc(kp[i], t[i])
    return (time.perf_counter() - t0) / len(t) * 1e3


def top_features(model, names, kind, k=8):
    clf = model[-1]
    if kind == "logreg":
        imp = np.abs(clf.coef_).mean(0)
    elif kind == "rf":
        imp = clf.feature_importances_
    else:
        return None
    o = np.argsort(imp)[::-1][:k]
    return [{"feature": names[i], "importance": float(imp[i])} for i in o]


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    sessions = get_sessions(args)
    if len(sessions) < 2:
        raise SystemExit("Cần >= 2 session.")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pres = [prepare(s, args.win) for s in sessions]
    rows = {}

    yt, yp, lat, fa, _ = loso(pres, tune)
    r = report(yt, yp)
    rows["formula"] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "flap_f1": r["per_class"]["flap"]["f1"],
                       "per_class_f1": {c: r["per_class"][c]["f1"] for c in CLASSES},
                       "idle_fa_rate": fa_summary(fa)["fa_rate"], "latency": lat_summary(lat),
                       "infer_ms": float(np.mean([formula_ms(s) for s in sessions[:2]])), "train_s": 0.0,
                       "explain": EXPLAIN["formula"]}
    for kind in MODELS:
        yt, yp, lat, fa, one_ms, tr_s, (m, names, _) = ml_loso(sessions, pres, args.win, kind)
        r = report(yt, yp)
        rows[kind] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "flap_f1": r["per_class"]["flap"]["f1"],
                      "per_class_f1": {c: r["per_class"][c]["f1"] for c in CLASSES},
                      "idle_fa_rate": fa_summary(fa)["fa_rate"], "latency": lat_summary(lat), "infer_ms": one_ms,
                      "train_s": tr_s, "explain": EXPLAIN[kind], "top_features": top_features(m, names, kind)}
        print(f"  {kind}: macroF1={r['macro_f1']:.3f}")

    keys = ["formula"] + MODELS
    md = ["| Phương pháp | Acc | macro-F1 | F1 flap | Báo nhầm idle | Trễ flap (từ tay đập, ms, trung vị) | Suy luận (ms/mẫu) | Giải thích |",
          "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        d = rows[k]; lt = d["latency"]["from_onset_ms"]
        md.append(f"| {NICE[k]} | {d['accuracy']:.3f} | {d['macro_f1']:.3f} | {d['flap_f1']:.3f} | "
                  f"{d['idle_fa_rate']:.3f} | {lt['median'] if lt else float('nan'):.0f} | {d['infer_ms']:.3f} | {d['explain']} |")
    (out / "ml_vs_formula.md").write_text(encoding='utf-8', data="\n".join(md) + "\n")
    (out / "ml_vs_formula.json").write_text(encoding='utf-8', data=json.dumps({"synthetic": bool(args.synthetic), "rows": rows}, indent=2,
                                                       ensure_ascii=False))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(keys)); lab = [NICE[k].split(" (")[0] for k in keys]
    ax[0].bar(x, [rows[k]["macro_f1"] for k in keys], color=["tab:red"] + ["tab:blue"] * 4)
    ax[0].set_ylim(min(0.5, min(rows[k]["macro_f1"] for k in keys) - .05), 1.02); ax[0].set_title("macro-F1 (LOSO)")
    ax[1].bar(x, [(rows[k]["latency"]["from_onset_ms"] or {"median": np.nan})["median"] for k in keys],
              color=["tab:red"] + ["tab:blue"] * 4); ax[1].set_title("Độ trễ phát hiện flap (ms, từ lúc tay đập xuống)")
    ax[2].bar(x, [rows[k]["infer_ms"] for k in keys], color=["tab:red"] + ["tab:blue"] * 4)
    ax[2].set_yscale("log"); ax[2].set_title("Thời gian suy luận 1 mẫu (ms, log)")
    for a in ax:
        a.set_xticks(x, lab, rotation=20, ha="right"); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "ml_vs_formula.png", dpi=130); plt.close(fig)
    print("\n".join(md)); print(f"\nĐã ghi {out}/ml_vs_formula.{{json,png,md}}")


if __name__ == "__main__":
    main()
