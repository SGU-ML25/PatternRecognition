"""[P2] PCA / t-SNE trên vector đặc trưng đã gán nhãn — các lớp động tác có tách được không?

  python -m analysis.embed_features --data data/sessions
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CLASSES, add_data_args, get_sessions  # noqa: E402
from .features import stack_windows  # noqa: E402

COLORS = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    sessions = get_sessions(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, y, g, d, names = stack_windows(sessions, "engineered", args.win, stride=3)
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
    pca = PCA(2).fit(Z); P = pca.transform(Z)
    T = TSNE(2, perplexity=min(30, max(5, len(Z) // 50)), init="pca", random_state=0).fit_transform(Z)
    sil = float(silhouette_score(Z, y))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, E, ttl in [(ax[0], P, f"PCA (2 thành phần giải thích {pca.explained_variance_ratio_.sum():.0%} phương sai)"),
                      (ax[1], T, "t-SNE")]:
        for c in range(5):
            m = y == c
            a.scatter(E[m, 0], E[m, 1], s=8, c=COLORS[c], label=CLASSES[c], alpha=.7)
        a.set_title(ttl); a.grid(alpha=.3)
    ax[0].legend(markerscale=3)
    fig.suptitle(f"Vector đặc trưng cửa sổ 0.5s theo lớp động tác — silhouette (không gian gốc) = {sil:.2f}")
    fig.tight_layout(); fig.savefig(out / "embedding.png", dpi=130); plt.close(fig)
    (out / "embedding.json").write_text(encoding='utf-8', data=json.dumps({"synthetic": bool(args.synthetic), "silhouette": sil,
                                                    "pca_var": pca.explained_variance_ratio_.tolist(),
                                                    "n_windows": int(len(y))}, indent=2))
    print(f"silhouette={sil:.3f}; đã ghi {out}/embedding.{{png,json}}")


if __name__ == "__main__":
    main()
