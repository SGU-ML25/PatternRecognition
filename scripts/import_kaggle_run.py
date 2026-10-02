"""Nhập kết quả một lần train trên Kaggle (file zip tải từ tab Output) vào runs/ để lưu trữ.

    python scripts/import_kaggle_run.py ~/Downloads/skypose_20261002-2100_resnet50_kaggle.zip

Kết quả: runs/<tên lần chạy>/ gồm run_info.json, logs/, <backbone>/train.log, train_log.csv,
eval_*.json, figures/ và <backbone>/best.pt. Log + biểu đồ có thể commit lên git
(*.pt bị .gitignore loại trừ vì nặng).
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="ghi đè nếu lần chạy đã tồn tại")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    for zp in args.zips:
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            run_name = names[0].split("/")[0]
            dest = RUNS / run_name
            if dest.exists() and not args.force:
                print(f"Bỏ qua {zp.name}: {dest} đã tồn tại (dùng --force để ghi đè)")
                continue
            z.extractall(RUNS)
        info = json.loads((dest / "run_info.json").read_text())
        print(f"\n== {run_name}  [{info.get('status')}]")
        git = info.get("git", {})
        print(f"   commit  : {git.get('commit', '?')[:8]} {git.get('message', '')}")
        print(f"   GPU     : {info.get('gpus')}")
        for s in info.get("steps", []):
            print(f"   {s['step']:<17} {'ok ' if s.get('ok') else 'LỖI'} {s.get('minutes', '?'):>7} phút")
        evals = list(dest.glob("eval_*.json"))
        if evals:
            ev = json.loads(evals[0].read_text())
            print(f"   COCO AP : {ev['coco']['AP'] * 100:.1f}%  (AP50 {ev['coco']['AP50'] * 100:.1f}%)  "
                  f"epoch {ev.get('epoch')}")
        for best in dest.glob("*/best.pt"):
            print(f"   model   : {best.relative_to(ROOT)}")
            print(f"   chạy game: SKYPOSE_CKPT={best.relative_to(ROOT)} .venv/bin/uvicorn server.app:app --port 8765")


if __name__ == "__main__":
    main()
