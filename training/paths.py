"""Đường dẫn dữ liệu dùng chung. Mặc định theo cấu trúc repo; đổi bằng biến môi trường
(vd. trên Kaggle: dataset nằm ở /kaggle/input, chỉ đọc)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# thư mục chứa person_keypoints_{train,val}2017.json
ANN_DIR = Path(os.environ.get("SKYPOSE_ANN_DIR", ROOT / "data" / "coco" / "annotations"))
# thư mục crop 320px + {split}_index.npz do prepare_data.py tạo ra
CROPS_DIR = Path(os.environ.get("SKYPOSE_CROPS", ROOT / "data" / "crops"))
# ảnh val2017 mặc định (nếu đã tải zip về máy)
DEFAULT_VAL_IMAGES = ROOT / "data" / "coco" / "val2017"
