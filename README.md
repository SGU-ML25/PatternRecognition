# SkyPose — Điều khiển chim bay bằng tư thế cơ thể

Đồ án môn **Phân tích và Nhận dạng mẫu**. Người chơi đứng trước webcam, dang tay như cánh chim
để lái con chim bay qua 16 cổng trong thế giới 3D. Khung xương được nhận dạng bởi **model
pose estimation tự train trên COCO-Keypoints** (SimpleBaseline, ResNet50).

Xem `PLAN.md` (kế hoạch) và `report/` (báo cáo).

## Cài đặt

```bash
python3 -m venv .venv --system-site-packages   # dùng torch/torchvision có sẵn (CUDA)
.venv/bin/pip install -r requirements.txt
```

## 1. Chuẩn bị dữ liệu

```bash
# annotation + val2017 (~1 GB)
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
cd ../..
# cắt vùng người -> data/crops/  (train tải lẻ từng ảnh, không cần tải 18 GB train2017.zip)
.venv/bin/python training/prepare_data.py --split val
.venv/bin/python training/prepare_data.py --split train
```

## 2. Huấn luyện & đánh giá

```bash
cd training
../.venv/bin/python train.py --backbone resnet50 --epochs 20 --out ../checkpoints/resnet50
../.venv/bin/python evaluate.py --ckpt ../checkpoints/resnet50/best.pt --out ../report/eval_resnet50.json
```

### Train trên Kaggle (GPU miễn phí)

`kaggle/skypose_train_kaggle.ipynb` clone repo này từ GitHub rồi gọi đúng các script trong
`training/` (dữ liệu và output được trỏ qua biến môi trường `SKYPOSE_ANN_DIR`, `SKYPOSE_CROPS`
— xem `training/paths.py`). Các bước: Import notebook → GPU T4 x2 + Internet On → (tuỳ chọn) gắn
dataset COCO 2017 → sửa `REPO_URL` → Save & Run All. Kết quả: `out/skypose_<backbone>.zip`.

Sửa notebook: sửa `kaggle/skypose_train_kaggle.py` rồi chạy `python kaggle/build_notebook.py`.

## 3. Chạy game

```bash
SKYPOSE_CKPT=checkpoints/resnet50/best.pt .venv/bin/uvicorn server.app:app --port 8765
```

- Game: <http://localhost:8765/> (webcam cần `localhost` hoặc HTTPS)
- API docs (Swagger): <http://localhost:8765/docs>

| Endpoint | Mô tả |
|---|---|
| `POST /api/predict` | ảnh → 17 khớp + lệnh điều khiển (JSON) |
| `POST /api/predict/image` | ảnh → ảnh có vẽ skeleton |
| `WS /ws/pose` | luồng JPEG → JSON thời gian thực (game dùng) |
| `GET /api/model/info`, `/api/health` | thông tin model |

## Điều khiển

| Động tác | Chim | Bàn phím |
|---|---|---|
| Dang ngang 2 tay | lượn | — |
| Nghiêng tay như cánh máy bay | rẽ trái/phải | ← → |
| Vẫy mạnh tay xuống | vỗ cánh, bay lên | Space |
| Khép tay sát người | bổ nhào | ↓ |

## Cấu trúc

```
training/   prepare_data, dataset, model (SimpleBaseline), train, evaluate, make_figures
server/     app.py (FastAPI), pose_engine.py (tracker + One-Euro), controls.py (toán điều khiển)
web/        index.html, main.js (game loop + vật lý), world.js, bird.js, pose.js (webcam client)
report/     báo cáo + biểu đồ
```
