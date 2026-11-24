"""[P1] Lọc tín hiệu: không lọc vs EMA vs One-Euro vs Kalman — đánh đổi jitter ↔ độ trễ.

  python -m analysis.compare_filters --data data/sessions

Dùng `kp_raw` (khớp THÔ trước lọc) lưu trong session, áp từng bộ lọc offline (cùng cách server làm: lọc trên
toạ độ chuẩn hoá /max(h,w)), rồi đo trên 2 tín hiệu điều khiển: elev (đơn vị S, dùng để vỗ cánh) và roll (độ, để rẽ).
  - JITTER   = std của hiệu liên tiếp / √2 trong các đoạn ĐỨNG YÊN (nhãn idle)  — càng thấp càng mượt.
  - ĐỘ TRỄ   = độ dịch thời gian τ làm MSE(tín hiệu lọc(t), tín hiệu thô(t−τ)) nhỏ nhất, tính trên các đoạn
               tay đang chuyển động nhanh (không cần nhãn). Tín hiệu thô là tham chiếu không trễ.
Mỗi họ bộ lọc được quét tham số -> đường đánh đổi; sao = cấu hình đang dùng (One-Euro min_cutoff=1, beta=3).
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import IDLE, add_data_args, get_sessions  # noqa: E402
from .features import frame_features  # noqa: E402
from .filters import apply_filter  # noqa: E402

CONFIGS = ([("none", {})]
           + [("ema", {"alpha": a}) for a in (0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1)]
           + [("oneeuro", {"min_cutoff": mc, "beta": b}) for mc in (0.3, 1.0, 3.0) for b in (0.0, 3.0, 10.0)]
           + [("kalman", {"q": q}) for q in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)])
DEPLOYED = ("oneeuro", {"min_cutoff": 1.0, "beta": 3.0})
GRID_HZ, MAX_LAG_S, WIN_S = 120.0, 0.30, 2.0
MOVE_THR = {"elev": 0.6, "roll": 12.0}  # biên độ (S hoặc độ) trong cửa sổ 2s để coi là 'đang chuyển động'


def signals(kp):
    F = frame_features(kp)
    return {"elev": F["elev"], "roll": np.degrees(F["roll"])}


def jitter(x, label):
    m = (label == IDLE)
    ok = m[1:] & m[:-1] & ~np.isnan(x[1:]) & ~np.isnan(x[:-1])
    d = (x[1:] - x[:-1])[ok]
    return float(d.std() / np.sqrt(2)) if len(d) > 10 else float("nan")


def lag_ms(f, r, t, name):
    ok = ~np.isnan(f) & ~np.isnan(r)
    if ok.sum() < 30:
        return float("nan")
    grid = np.arange(t[0], t[-1], 1 / GRID_HZ)
    fi, ri = np.interp(grid, t[ok], f[ok]), np.interp(grid, t[ok], r[ok])
    n, K = int(WIN_S * GRID_HZ), int(MAX_LAG_S * GRID_HZ)
    sm = np.convolve(ri, np.ones(9) / 9, mode="same")
    mse, cnt = np.zeros(K + 1), 0
    for a in range(0, len(grid) - n, n // 2):
        if np.ptp(sm[a:a + n]) < MOVE_THR[name]:
            continue
        for k in range(K + 1):
            mse[k] += np.mean((fi[a + k:a + n] - ri[a:a + n - k]) ** 2)
        cnt += 1
    return float(np.argmin(mse) / GRID_HZ * 1e3) if cnt else float("nan")


def pareto(points):
    """points: [(jitter, lag)] -> mảng bool điểm không bị trội."""
    P = np.array(points); keep = np.ones(len(P), bool)
    for i, p in enumerate(P):
        keep[i] = not np.any((P[:, 0] <= p[0]) & (P[:, 1] <= p[1]) & ((P[:, 0] < p[0]) | (P[:, 1] < p[1])))
    return keep


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    sessions = get_sessions(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, kw in CONFIGS:
        J = {"elev": [], "roll": []}; L = {"elev": [], "roll": []}
        for s in sessions:
            kpf = apply_filter(s["kp_raw"], s["t"], s["wh"], kind, **kw)
            sf, sr = signals(kpf), signals(s["kp_raw"])
            for k in ("elev", "roll"):
                J[k].append(jitter(sf[k], s["label"])); L[k].append(lag_ms(sf[k], sr[k], s["t"], k))
        rows.append({"filter": kind, "params": kw, **{f"jitter_{k}": float(np.nanmean(J[k])) for k in J},
                     **{f"lag_{k}_ms": float(np.nanmean(L[k])) for k in L}})
        print(f"  {kind:8s} {kw}  jitter_elev={rows[-1]['jitter_elev']:.4f}  lag_elev={rows[-1]['lag_elev_ms']:.0f}ms")
    pf = pareto([(r["jitter_elev"], r["lag_elev_ms"]) for r in rows])
    for r, p in zip(rows, pf):
        r["pareto_elev"] = bool(p)

    colors = {"none": "k", "ema": "tab:blue", "oneeuro": "tab:green", "kalman": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, sig, unit in [(ax[0], "elev", "S"), (ax[1], "roll", "°")]:
        for fam in ["ema", "kalman"]:
            R = sorted([r for r in rows if r["filter"] == fam], key=lambda r: r[f"jitter_{sig}"])
            a.plot([r[f"jitter_{sig}"] for r in R], [r[f"lag_{sig}_ms"] for r in R], "o-", c=colors[fam], label=fam, alpha=.8)
        for i, mc in enumerate(sorted({r["params"]["min_cutoff"] for r in rows if r["filter"] == "oneeuro"})):
            R = sorted([r for r in rows if r["filter"] == "oneeuro" and r["params"]["min_cutoff"] == mc],
                       key=lambda r: r["params"]["beta"])  # mỗi đường: 1 min_cutoff, beta tăng dần (0, 3, 10)
            a.plot([r[f"jitter_{sig}"] for r in R], [r[f"lag_{sig}_ms"] for r in R], "o-", c=colors["oneeuro"],
                   alpha=.45 + .2 * i, label="oneeuro (mỗi đường 1 min_cutoff, β↑)" if i == 0 else None)
            a.annotate(f"fc={mc:g}", (R[0][f"jitter_{sig}"], R[0][f"lag_{sig}_ms"]), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
        r0 = next(r for r in rows if r["filter"] == "none")
        a.plot(r0[f"jitter_{sig}"], r0[f"lag_{sig}_ms"], "s", c="k", ms=9, label="không lọc")
        rd = next(r for r in rows if (r["filter"], r["params"]) == DEPLOYED)
        a.plot(rd[f"jitter_{sig}"], rd[f"lag_{sig}_ms"], "*", c="gold", mec="k", ms=18, label="đang dùng (One-Euro 1, 3)")
        a.set_xlabel(f"jitter khi đứng yên ({unit}, thấp = mượt)"); a.set_ylabel("độ trễ (ms, thấp = nhanh)")
        a.set_title(f"Tín hiệu {sig}"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Đánh đổi jitter ↔ độ trễ (góc dưới-trái là tốt nhất)"); fig.tight_layout()
    fig.savefig(out / "filters_tradeoff.png", dpi=130); plt.close(fig)

    md = ["| Bộ lọc | Tham số | Jitter elev (S) | Trễ elev (ms) | Jitter roll (°) | Trễ roll (ms) | Pareto(elev) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['filter']} | {r['params']} | {r['jitter_elev']:.4f} | {r['lag_elev_ms']:.0f} | "
                  f"{r['jitter_roll']:.3f} | {r['lag_roll_ms']:.0f} | {'✓' if r['pareto_elev'] else ''} |")
    (out / "filters_tradeoff.md").write_text(encoding='utf-8', data="\n".join(md) + "\n")
    (out / "filters_tradeoff.json").write_text(encoding='utf-8', data=json.dumps({"synthetic": bool(args.synthetic), "rows": rows}, indent=2))
    print(f"\nĐã ghi {out}/filters_tradeoff.{{json,png,md}}")


if __name__ == "__main__":
    main()
