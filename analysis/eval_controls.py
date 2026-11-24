"""[P0] Đánh giá bộ điều khiển công thức (server/controls.py) trên dữ liệu webcam có nhãn.

  python -m analysis.eval_controls --data data/sessions

Quy trình:
  1. Với mỗi khung tính quyết định theo cửa sổ nhân quả 0.5s: flap / dive / left / right / idle.
  2. ĐỘ CHÍNH XÁC theo lệnh: precision / recall / F1 / ma trận nhầm lẫn trên các cửa sổ 'sạch' (cùng 1 nhãn).
  3. ĐỘ TRỄ phát hiện flap: (a) từ lúc nhãn bắt đầu, (b) từ lúc tay thực sự bắt đầu đập xuống (suy ra từ
     khớp THÔ, nên không tính thời gian phản xạ của người quay).
  4. TỈ LỆ BÁO NHẦM khi đứng yên.
  5. NGƯỠNG chọn bằng lưới (F1 một-đấu-tất-cả) + ROC, kiểm bằng leave-one-session-out: ngưỡng học trên
     các session còn lại, đánh giá trên session bị giữ ra. So với ngưỡng 'đoán' mặc định.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import (CLASSES, DIVE, FLAP, IDLE, LEFT, RIGHT, add_data_args, get_sessions,  # noqa: E402
                     label_purity, roll_agg)
from .features import ControlParams, dive_score, flap_track, flap_velocity, frame_features  # noqa: E402

FLAP_OUT_THR = 0.1  # `flap` (0..1) > ngưỡng này coi là đang vỗ  (<=> v > flap_v0 + 0.4 S/s)


@dataclass
class Rule:
    """Quy tắc quyết định trên đặc trưng. Ngưỡng mặc định = suy ra từ hằng số trong controls.py."""
    roll_thr_deg: float = 15.0   # |roll| > ngưỡng -> rẽ   (controls: bank = roll/40°)
    flap_v0: float = 1.0         # S/giây
    flap_spread: float = 1.8
    dive_spread: float = 1.6     # dive>0.5 <=> spread<1.6 khi below>=0.5
    dive_below: float = 0.5


DEFAULT = Rule()
GRIDS = {
    "roll": np.arange(4.0, 41.0, 1.0),
    "flap_v0": np.arange(0.5, 3.01, 0.25),
    "flap_spread": np.array([1.2, 1.5, 1.8, 2.1, 2.4, 2.7]),
    "dive_spread": np.arange(1.4, 3.01, 0.2),
    "dive_below": np.arange(0.0, 1.01, 0.1),
}


# ---------------------------------------------------------------- tiền tính cho một session
def prepare(sess, win):
    t, F = sess["t"], frame_features(sess["kp"])
    v = flap_velocity(F, t)
    Fr = frame_features(sess["kp_raw"])
    pre = {"sess": sess, "F": F, "v": v, "t": t, "win": win, "_flap_cache": {},
           "v_raw": flap_velocity(Fr, t), "elev_raw": Fr["elev"],
           "roll_med": np.nan_to_num(roll_agg(np.degrees(F["roll"]), t, win, "median")),
           "spread_med": roll_agg(F["spread"], t, win, "median"),
           "below_med": roll_agg(F["below"], t, win, "median"),
           "pure": label_purity(sess["label"], t, win), "y": sess["label"]}
    pre["dive_med"] = roll_agg(dive_score(F["spread"], F["below"], ControlParams()), t, win, "median")
    return pre


def flap_window(pre, v0, spread_thr):
    key = (round(v0, 3), round(spread_thr, 3))
    if key not in pre["_flap_cache"]:
        F = pre["F"]
        f = flap_track(pre["v"], F["spread"], F["sh_ok"], F["tips_ok"], v0=v0, spread_thr=spread_thr)
        pre["_flap_cache"][key] = np.nan_to_num(roll_agg(f, pre["t"], pre["win"], "max", 1))
    return pre["_flap_cache"][key]


def predict(pre, r: Rule):
    """Nhãn dự đoán cho từng khung. Thứ tự ưu tiên: flap > dive > rẽ > idle."""
    fl = flap_window(pre, r.flap_v0, r.flap_spread) > FLAP_OUT_THR
    dv = (pre["spread_med"] < r.dive_spread) & (pre["below_med"] > r.dive_below)
    rl = pre["roll_med"]
    pred = np.full(len(rl), IDLE)
    pred[rl > r.roll_thr_deg] = LEFT
    pred[rl < -r.roll_thr_deg] = RIGHT
    pred[np.nan_to_num(dv.astype(float)) > 0] = DIVE
    pred[fl] = FLAP
    return pred


# ---------------------------------------------------------------- chọn ngưỡng
def _f1(y, pos, pred_pos):
    tp = np.sum((y == pos) & pred_pos); fp = np.sum((y != pos) & pred_pos); fn = np.sum((y == pos) & ~pred_pos)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _cat(pres, key):
    return np.concatenate([p[key][p["pure"]] for p in pres])


def plateau_pick(cands, axes, tol=0.005):
    """cands: [(chỉ số trên từng trục lưới, điểm F1)]. Lấy mọi ứng viên có điểm >= max - tol rồi chọn cái gần
    TÂM của 'bình nguyên' đó — tránh chọn ngưỡng sát mép lưới khi nhiều ngưỡng cho kết quả ngang nhau."""
    best = max(sc for _, sc in cands)
    pl = np.array([ix for ix, sc in cands if sc >= best - tol], float)
    c = pl.mean(0)
    k = int(np.argmin(((pl - c) ** 2).sum(1)))
    ix = tuple(int(i) for i in pl[k])
    return tuple(axes[d][i] for d, i in enumerate(ix)), best


def tune(pres, warn=False):
    """Chọn từng nhóm ngưỡng bằng F1 một-đấu-tất-cả trên các cửa sổ sạch của các session `pres`."""
    y = _cat(pres, "y")
    roll = _cat(pres, "roll_med")
    g = GRIDS["roll"]
    (roll_thr,), _ = plateau_pick(
        [((i,), (_f1(y, LEFT, roll > th) + _f1(y, RIGHT, roll < -th)) / 2) for i, th in enumerate(g)], [g])
    gv, gs = GRIDS["flap_v0"], GRIDS["flap_spread"]
    cands = []
    for i, v0 in enumerate(gv):
        for j, sp in enumerate(gs):
            fw = np.concatenate([flap_window(p, v0, sp)[p["pure"]] for p in pres]) > FLAP_OUT_THR
            cands.append(((i, j), _f1(y, FLAP, fw)))
    (v0, sp), _ = plateau_pick(cands, [gv, gs])
    spm, blm = _cat(pres, "spread_med"), _cat(pres, "below_med")
    gd, gb = GRIDS["dive_spread"], GRIDS["dive_below"]
    cands = [((i, j), _f1(y, DIVE, (spm < sd) & (blm > bt))) for i, sd in enumerate(gd) for j, bt in enumerate(gb)]
    (sd, bt), _ = plateau_pick(cands, [gd, gb])
    rule = Rule(float(roll_thr), float(v0), float(sp), float(sd), float(bt))
    if warn:
        for name, val, grid in [("roll_thr_deg", rule.roll_thr_deg, g), ("flap_v0", rule.flap_v0, gv),
                                ("flap_spread", rule.flap_spread, gs), ("dive_spread", rule.dive_spread, gd),
                                ("dive_below", rule.dive_below, gb)]:
            if val in (grid[0], grid[-1]):
                print(f"   !! {name}={val:.2f} nằm sát mép lưới [{grid[0]:.2f}, {grid[-1]:.2f}] — "
                      f"cân nhắc mở rộng GRIDS['...'] rồi chạy lại")
    return rule


# ---------------------------------------------------------------- chỉ số
def seg_bounds(label, value):
    m = np.r_[False, label == value, False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def flap_latencies(pre, pred, v_onset=1.0):
    """Độ trễ phát hiện flap (giây) cho từng đoạn flap: từ nhãn, và từ lúc tay thực sự đập xuống."""
    t, lab = pre["t"], pre["y"]
    rows = []
    for a, b in seg_bounds(lab, FLAP):
        det = np.where(pred[a:b + 1] == FLAP)[0]
        t_det = t[a + det[0]] if len(det) else np.nan
        on = np.where(np.nan_to_num(pre["v_raw"][a:b + 1]) > v_onset)[0]
        t_on = t[a + on[0]] if len(on) else np.nan
        rows.append({"t_start": float(t[a]), "from_label": float(t_det - t[a]), "from_onset": float(t_det - t_on),
                     "detected": bool(len(det))})
    return rows


def idle_false_alarm(pre, pred):
    m = pre["pure"] & (pre["y"] == IDLE)
    if not m.any():
        return {"idle_windows": 0, "fa_rate": float("nan"), "fa_per_min": float("nan")}
    bad = pred[m] != IDLE
    # sự kiện báo nhầm = số lần chuyển từ idle-đúng sang idle-sai (đếm theo cụm liên tiếp)
    ev = int(np.sum(np.diff(np.r_[0, bad.astype(int)]) == 1))
    mins = m.sum() / max(1.0, 1.0 / np.median(np.diff(pre["t"]))) / 60.0
    return {"idle_windows": int(m.sum()), "fa_rate": float(bad.mean()), "fa_per_min": float(ev / mins)}


def report(y_true, y_pred):
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    return {"accuracy": float((y_true == y_pred).mean()), "macro_f1": float(f.mean()),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
                          for i, c in enumerate(CLASSES)}, "confusion": cm.tolist()}


def loso(pres, rule_fn):
    """rule_fn(train_pres) -> Rule. Trả (y_true, y_pred, latencies, false_alarm, rules)."""
    yt, yp, lat, fa, rules = [], [], [], [], []
    for k, te in enumerate(pres):
        rule = rule_fn([p for j, p in enumerate(pres) if j != k])
        pr = predict(te, rule)
        yt.append(te["y"][te["pure"]]); yp.append(pr[te["pure"]])
        lat += flap_latencies(te, pr); fa.append(idle_false_alarm(te, pr)); rules.append(asdict(rule))
    return np.concatenate(yt), np.concatenate(yp), lat, fa, rules


def lat_summary(lat):
    d = [x for x in lat if x["detected"]]
    f = lambda k: np.array([x[k] for x in d])  # noqa: E731
    out = {"n_segments": len(lat), "missed": len(lat) - len(d)}
    for k in ("from_label", "from_onset"):
        a = f(k)
        a = a[~np.isnan(a)] if len(a) else a
        out[k + "_ms"] = ({"median": float(np.median(a) * 1e3), "p25": float(np.percentile(a, 25) * 1e3),
                           "p75": float(np.percentile(a, 75) * 1e3), "mean": float(a.mean() * 1e3)} if len(a) else None)
    return out


def fa_summary(fa):
    n = sum(x["idle_windows"] for x in fa)
    return {"fa_rate": float(sum(x["fa_rate"] * x["idle_windows"] for x in fa if x["idle_windows"]) / max(n, 1)),
            "fa_per_min": float(np.nanmean([x["fa_per_min"] for x in fa]))}


# ---------------------------------------------------------------- vẽ
def plot_all(out, pres, results, final_rule):
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for k, name in enumerate(["default", "tuned"]):
        cm = np.array(results[name]["confusion"], float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a = ax[0, k]
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(5), CLASSES); a.set_yticks(range(5), CLASSES)
        a.set_xlabel("dự đoán"); a.set_ylabel("nhãn thật")
        a.set_title(f"{'Ngưỡng đoán' if name == 'default' else 'Ngưỡng chọn bằng lưới'} (LOSO)\n"
                    f"acc={results[name]['accuracy']:.3f}  macroF1={results[name]['macro_f1']:.3f}")
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                       color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    y = _cat(pres, "y")
    sc = {LEFT: (_cat(pres, "roll_med"), "left: roll trung vị (°)", final_rule.roll_thr_deg, DEFAULT.roll_thr_deg),
          RIGHT: (-_cat(pres, "roll_med"), "right: -roll trung vị (°)", final_rule.roll_thr_deg, DEFAULT.roll_thr_deg),
          FLAP: (np.concatenate([flap_window(p, 1.0, 1.8)[p["pure"]] for p in pres]), "flap: max flap(0..1) trong 0.5s",
                 FLAP_OUT_THR, FLAP_OUT_THR),
          DIVE: (_cat(pres, "dive_med"), "dive: điểm dive trung vị", None, None)}
    for (cls, (s, title, thr_t, thr_d)), a in zip(sc.items(), [ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1]]):
        pos = (y == cls)
        if pos.sum() == 0 or (~pos).sum() == 0:
            a.set_title(f"{CLASSES[cls]}: thiếu dữ liệu"); continue
        s_clean = np.nan_to_num(s, nan=0.0)
        fpr, tpr, _ = roc_curve(pos, s_clean)
        a.plot(fpr, tpr, lw=2); a.plot([0, 1], [0, 1], "k:", lw=1)
        a.set_title(f"ROC {title}\nAUC={roc_auc_score(pos, s_clean):.3f}", fontsize=10)
        a.set_xlabel("FPR"); a.set_ylabel("TPR")
        for thr, c, lb in [(thr_d, "tab:red", "đoán"), (thr_t, "tab:green", "chọn")]:
            if thr is None:
                continue
            a.plot((s[~pos] > thr).mean(), (s[pos] > thr).mean(), "o", c=c, ms=9, label=lb)
        if cls != DIVE:
            a.legend()
    lat = results["tuned"]["latency"]
    a = ax[1, 2]
    for key, c in [("from_label", "tab:blue"), ("from_onset", "tab:orange")]:
        vals = [x[key] * 1e3 for x in results["_lat_tuned"] if x["detected"] and not np.isnan(x[key])]
        if vals:
            a.hist(vals, bins=15, alpha=.6, color=c, label={"from_label": "từ lúc nhãn bắt đầu",
                                                            "from_onset": "từ lúc tay đập xuống"}[key])
    a.set_xlabel("độ trễ phát hiện flap (ms)"); a.set_ylabel("số đoạn"); a.legend()
    a.set_title(f"Độ trễ flap — bỏ sót {lat['missed']}/{lat['n_segments']} đoạn")
    ax[1, 3].axis("off")
    ax[1, 3].text(0, 1, "Ngưỡng đề xuất (huấn luyện trên TẤT CẢ session):\n\n" + "\n".join(
        f"{k:12s} = {v:.2f}   (đoán: {getattr(DEFAULT, k):.2f})" for k, v in asdict(final_rule).items())
                  + f"\n\nflap_out_thr = {FLAP_OUT_THR}", va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(out) / "controls_eval.png", dpi=130)
    plt.close(fig)


def to_markdown(res):
    L = ["| Lệnh | Precision (đoán) | Recall (đoán) | Precision (chọn) | Recall (chọn) | F1 (đoán) | F1 (chọn) |", "|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        d, t = res["default"]["per_class"][c], res["tuned"]["per_class"][c]
        L.append(f"| {c} | {d['precision']:.3f} | {d['recall']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
                 f"{d['f1']:.3f} | {t['f1']:.3f} |")
    L.append(f"| **tổng** | | | | | macro {res['default']['macro_f1']:.3f} | macro {res['tuned']['macro_f1']:.3f} |")
    return "\n".join(L)


def main():
    ap = add_data_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    args = ap.parse_args()
    sessions = get_sessions(args)
    if len(sessions) < 2:
        raise SystemExit("Cần >= 2 session để kiểm leave-one-session-out (ngưỡng học trên session khác).")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pres = [prepare(s, args.win) for s in sessions]
    print(f"{len(pres)} session, {sum(p['pure'].sum() for p in pres)} cửa sổ sạch")

    res = {}
    for name, fn in [("default", lambda tr: DEFAULT), ("tuned", tune)]:
        yt, yp, lat, fa, rules = loso(pres, fn)
        res[name] = report(yt, yp)
        res[name]["latency"] = lat_summary(lat); res[name]["idle_false_alarm"] = fa_summary(fa)
        res[name]["rules_per_fold"] = rules; res["_lat_" + name] = lat
        print(f"\n== {name}: acc={res[name]['accuracy']:.3f}  macroF1={res[name]['macro_f1']:.3f}")
        for c, d in res[name]["per_class"].items():
            print(f"   {c:6s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (n={d['n']})")
        print("   latency:", json.dumps(res[name]["latency"]))
        print("   idle báo nhầm:", res[name]["idle_false_alarm"])
    final = tune(pres, warn=True)
    print("\nNgưỡng đề xuất (tune trên tất cả session):", asdict(final))
    plot_all(out, pres, res, final)
    lat_raw = {k: v for k, v in res.items() if k.startswith("_lat")}
    res["final_rule"] = asdict(final); res["default_rule"] = asdict(DEFAULT)
    for k in lat_raw:
        del res[k]
    res["sessions"] = [{"name": p["sess"]["name"], "distance": p["sess"]["distance"]} for p in pres]
    res["synthetic"] = bool(args.synthetic)
    (out / "controls_eval.json").write_text(encoding='utf-8', data=json.dumps(res, indent=2, ensure_ascii=False))
    (out / "controls_eval.md").write_text(encoding='utf-8', data=to_markdown(res) + "\n")
    print(f"\nĐã ghi {out}/controls_eval.{{json,png,md}}")


if __name__ == "__main__":
    main()
