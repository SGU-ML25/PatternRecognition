"""[P1] Feature selection: đặc trưng nào thực sự cần?

  python -m analysis.feature_selection --data data/sessions [--model rf]

Hai phép đo trên 7 đặc trưng (mỗi đặc trưng gồm 5 thống kê cửa sổ, xử lý chung một nhóm):
  - Ablation: bỏ nhóm đặc trưng, huấn luyện lại, đo macro-F1 giảm bao nhiêu (leave-one-session-out).
  - Permutation importance: xáo trộn nhóm đặc trưng trên tập test, đo macro-F1 giảm (không huấn luyện lại).
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import add_data_args, get_sessions  # noqa: E402
from .features import ENGINEERED, stack_windows  # noqa: E402
from .mlutil import macro_f1, make_model  # noqa: E402


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--model", default="rf", choices=["rf", "logreg", "svm", "mlp"])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    sessions = get_sessions(args)
    if len(sessions) < 2:
        raise SystemExit("Cần >= 2 session.")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, y, g, _, names = stack_windows(sessions, "engineered", args.win, stride=2)
    cols = {f: [i for i, n in enumerate(names) if n.split(":")[0] == f] for f in ENGINEERED}
    rng = np.random.default_rng(0)
    folds = np.unique(g)

    full = {k: make_model(args.model).fit(X[g != k], y[g != k]) for k in folds}  # mỗi fold fit đúng 1 lần

    def pooled(drop=None, permute=None):
        yt, yp = [], []
        for k in folds:
            tr, te = g != k, g == k
            Xt = X[te].copy()
            if drop is not None:  # ablation: phải huấn luyện lại không có nhóm đặc trưng này
                keep = [i for i in range(X.shape[1]) if i not in cols[drop]]
                m, Xt = make_model(args.model).fit(X[tr][:, keep], y[tr]), Xt[:, keep]
            else:
                m = full[k]
                if permute is not None:  # permutation: xáo trộn các hàng của nhóm cột, KHÔNG huấn luyện lại
                    Xt[:, cols[permute]] = Xt[rng.permutation(len(Xt))][:, cols[permute]]
            yt.append(y[te]); yp.append(m.predict(Xt))
        return macro_f1(np.concatenate(yt), np.concatenate(yp))

    base = pooled()
    print(f"macro-F1 đầy đủ 7 đặc trưng ({args.model}): {base:.3f}")
    res = {}
    for f in ENGINEERED:
        ab = pooled(drop=f)
        pm = float(np.mean([pooled(permute=f) for _ in range(args.reps)]))
        res[f] = {"ablation_drop": base - ab, "f1_without": ab, "permutation_drop": base - pm}
        print(f"  {f:7s} bỏ -> F1 {ab:.3f} (giảm {base - ab:+.3f}) | xáo trộn -> giảm {base - pm:+.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, k, ttl in [(ax[0], "ablation_drop", "Ablation: bỏ đặc trưng rồi huấn luyện lại"),
                      (ax[1], "permutation_drop", "Permutation: xáo trộn đặc trưng trên tập test")]:
        o = sorted(ENGINEERED, key=lambda f: res[f][k])
        a.barh(o, [res[f][k] for f in o], color="tab:blue"); a.axvline(0, c="k", lw=.8)
        a.set_xlabel("giảm macro-F1"); a.set_title(ttl, fontsize=10); a.grid(axis="x", alpha=.3)
    fig.suptitle(f"Độ quan trọng đặc trưng ({args.model}, leave-one-session-out, F1 đầy đủ = {base:.3f})")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)
    md = ["| Đặc trưng | F1 khi bỏ | Giảm F1 (ablation) | Giảm F1 (permutation) |", "|---|---|---|---|"]
    md += [f"| {f} | {res[f]['f1_without']:.3f} | {res[f]['ablation_drop']:+.3f} | {res[f]['permutation_drop']:+.3f} |"
           for f in sorted(ENGINEERED, key=lambda f: -res[f]["ablation_drop"])]
    (out / "feature_importance.md").write_text(encoding='utf-8', data="\n".join(md) + "\n")
    (out / "feature_importance.json").write_text(encoding='utf-8', data=json.dumps({"synthetic": bool(args.synthetic), "model": args.model,
                                                             "base_macro_f1": base, "features": res}, indent=2))
    print(f"\nĐã ghi {out}/feature_importance.{{json,png,md}}")


if __name__ == "__main__":
    main()
