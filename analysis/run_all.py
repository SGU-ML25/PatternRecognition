"""Chạy toàn bộ phân tích mục B theo thứ tự:  python -m analysis.run_all --data data/sessions
(thêm --synthetic để chạy thử bằng dữ liệu giả)."""
import argparse
import subprocess
import sys
import time

from .common import ROOT

STEPS = [("eval_controls", "[P0] đánh giá bộ điều khiển công thức + chọn ngưỡng"),
         ("eval_normalization", "[P0] chuẩn hoá theo S vs toạ độ thô (1m → 2m)"),
         ("compare_ml", "[P1] học máy vs công thức"),
         ("feature_selection", "[P1] độ quan trọng đặc trưng"),
         ("compare_filters", "[P1] One-Euro vs EMA vs Kalman"),
         ("embed_features", "[P2] PCA / t-SNE")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sessions"))
    ap.add_argument("--out", default=str(ROOT / "report" / "feature_eval"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--only", nargs="*", help="chỉ chạy các bước này (tên module)")
    a = ap.parse_args()
    extra = ["--synthetic"] if a.synthetic else ["--data", a.data]
    failed = []
    for mod, desc in STEPS:
        if a.only and mod not in a.only:
            continue
        print(f"\n{'=' * 78}\n{desc}   (analysis.{mod})\n{'=' * 78}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", f"analysis.{mod}", "--out", a.out, *extra], cwd=ROOT)
        print(f"-> {'OK' if r.returncode == 0 else 'LỖI'} ({time.time() - t0:.0f}s)")
        if r.returncode:
            failed.append(mod)
    print("\nKết quả:", a.out, "| lỗi:", failed or "không")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
