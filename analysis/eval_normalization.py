"""[P0] Đặc trưng chuẩn hoá theo S vs toạ độ pixel thô: có bất biến khi đổi khoảng cách tới camera không?

  python -m analysis.eval_normalization --data data/sessions

Cần session quay ở >= 2 khoảng cách (vd 1m và 2m; ghi bằng record_session.py --distance 1m / 2m).
Thí nghiệm: huấn luyện ở khoảng cách A, kiểm tra ở khoảng cách B (và ngược lại). CÙNG bộ phân loại, chỉ
khác cách biểu diễn đầu vào:
    raw_px        toạ độ pixel thô
    center_px     trừ trung điểm vai (chỉ bất biến vị trí)
    center_scale  trừ trung điểm vai + chia S (bất biến vị trí + tỉ lệ)
    engineered    7 đặc trưng điều khiển (roll, spread, wing, below, elev, v) — đã chuẩn hoá theo S
Chuẩn so sánh (trần): cùng khoảng cách, kiểm leave-one-session-out.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import add_data_args, get_sessions  # noqa: E402
from .features import VARIANTS, stack_windows  # noqa: E402
from .mlutil import macro_f1, make_model  # noqa: E402

CLFS = ["logreg", "rf"]


def scenarios(dists, groups, dist_arr):
    sc = []
    for a in dists:
        for b in dists:
            if a != b:
                sc.append((f"train {a:g}m → test {b:g}m", dist_arr == a, dist_arr == b, None))
    for d in dists:
        gs = np.unique(groups[dist_arr == d])
        if len(gs) >= 2:
            sc.append((f"cùng {d:g}m (LOSO)", None, None, d))
    return sc


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    sessions = get_sessions(args)
    dists = sorted({s["distance"] for s in sessions if not np.isnan(s["distance"])})
    if len(dists) < 2:
        raise SystemExit(f"Chỉ có khoảng cách {dists}. Cần quay ở >= 2 khoảng cách (--distance 1m / 2m).")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = {v: stack_windows(sessions, v, args.win) for v in VARIANTS}
    res = {}
    for v in VARIANTS:
        X, y, g, d, _ = data[v]
        for sname, tr, te, same in scenarios(dists, g, d):
            for clf in CLFS:
                if same is None:
                    m = make_model(clf).fit(X[tr], y[tr]); yt, yp = y[te], m.predict(X[te])
                else:
                    yt, yp = [], []
                    for k in np.unique(g[d == same]):
                        trm, tem = (d == same) & (g != k), g == k
                        m = make_model(clf).fit(X[trm], y[trm]); yt.append(y[tem]); yp.append(m.predict(X[tem]))
                    yt, yp = np.concatenate(yt), np.concatenate(yp)
                res.setdefault(sname, {}).setdefault(clf, {})[v] = {
                    "macro_f1": macro_f1(yt, yp), "accuracy": float((yt == yp).mean()), "n_test": int(len(yt))}
        print(f"  xong {v}")

    snames = list(res)
    fig, axes = plt.subplots(1, len(CLFS), figsize=(7.5 * len(CLFS), 4.8), sharey=True)
    w = 0.2
    for ax, clf in zip(np.atleast_1d(axes), CLFS):
        for i, v in enumerate(VARIANTS):
            vals = [res[s][clf][v]["macro_f1"] for s in snames]
            b = ax.bar(np.arange(len(snames)) + (i - 1.5) * w, vals, w, label=v)
            for r_, val in zip(b, vals):
                ax.text(r_.get_x() + r_.get_width() / 2, val + .01, f"{val:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(snames)), snames, rotation=15, ha="right", fontsize=8)
        ax.set_title({"logreg": "Logistic Regression", "rf": "Random Forest"}[clf]); ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("macro-F1 trên tập test"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Chuẩn hoá theo S giúp khi đổi khoảng cách camera?", y=1.0)
    fig.tight_layout(); fig.savefig(out / "normalization.png", dpi=130); plt.close(fig)

    md = ["| Kịch bản | Bộ phân loại | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for s in snames:
        for clf in CLFS:
            md.append(f"| {s} | {clf} | " + " | ".join(f"{res[s][clf][v]['macro_f1']:.3f}" for v in VARIANTS) + " |")
    (out / "normalization.md").write_text(encoding='utf-8', data="\n".join(md) + "\n")
    (out / "normalization.json").write_text(encoding='utf-8', data=json.dumps({"synthetic": bool(args.synthetic), "results": res}, indent=2,
                                                       ensure_ascii=False))
    print("\n".join(md)); print(f"\nĐã ghi {out}/normalization.{{json,png,md}}")


if __name__ == "__main__":
    main()
