# SkyPose — Presentation (1 file HTML)

`skypose-presentation.html` — mở bằng trình duyệt là xem được, chạy offline (đã nhúng mermaid, CSS, biểu đồ).
Bố cục dựa theo mẫu *Thuyết minh tự động đa ngôn ngữ* (CSS, modal sơ đồ zoom/pan được dùng lại nguyên).

| # | Chương |
|---|---|
| 00 | Bài toán & kiến trúc tổng quan |
| 01 | Dữ liệu COCO-Keypoints & tiền xử lý, augmentation |
| 02 | Mô hình SimpleBaseline (ResNet18 + 3 deconv) |
| 03 | Huấn luyện: laptop → Kaggle, diễn biến theo epoch |
| 04 | Kết quả trên COCO val2017 (AP, PCK từng khớp) |
| 05 | Suy luận thời gian thực: tracker, flip-test, One-Euro |
| 06 | Điều khiển bằng toán |
| 07 | API server (FastAPI) |
| 08 | Web game 3D |
| 09 | Luồng end-to-end |
| 10 | **Chiến lược so sánh** — có số đo thực nghiệm (`experiments/`) |
| 11 | **Hướng phát triển** (P0/P1/P2) |
| 12 | Kiểm thử & rủi ro |
| 13 | Lộ trình & nghiệm thu |
| PL | Technology stack & cách chạy |

18 sơ đồ mermaid (nút 📊 đầu mỗi chương).

## Dựng lại

```bash
python3 report/presentation/build/assemble.py
```

- `build/c*.html` — nội dung từng nhóm chương; `{{FIG:tên}}` được thay bằng ảnh base64 từ
  `runs/20261002-1405_resnet18_kaggle/figures/tên.png`
- `build/diagram_data.py` — định nghĩa sơ đồ
- `build/vendor/` — phần dùng lại từ file mẫu (CSS, modal, thư viện mermaid 11.14, viewer)

## Số liệu Chapter 10

`experiments/run_experiments.py` (chạy trong `training/`: `../.venv/bin/python ../report/presentation/experiments/run_experiments.py out.json`)
→ `experiments/results_20261013.json`: flip-test on/off, epoch 3 vs 20, 1 vs 2 lượt trên 500 ảnh val,
tốc độ ResNet18/34/50 × FP32/FP16/CPU trên RTX A2000 Laptop.
