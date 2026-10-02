# Đồ án: SkyPose — Điều khiển chim bay bằng tư thế cơ thể

Môn: Phân tích và Nhận dạng mẫu
Ý tưởng tham khảo: game *skypeck* (video Instagram) — người chơi dang tay như cánh chim trước webcam để lái con chim bay qua các cổng (gate) trong thế giới 3D.

## 1. Mục tiêu

| # | Thành phần | Mô tả | Trạng thái |
|---|---|---|---|
| 1 | **Model 1 — Pose Estimation (tự train)** | CNN nhận ảnh người → toạ độ 17 khớp (COCO format) | ⏳ |
| 2 | API server | FastAPI: REST `/predict` + WebSocket `/ws/pose`, chạy model trên GPU | ⏳ |
| 3 | Điều khiển bằng toán | Từ khớp → lệnh nghiêng / vỗ cánh / bổ nhào | ⏳ |
| 4 | Web game 3D | Three.js: chim, địa hình low-poly, cổng, HUD, khung webcam có skeleton | ⏳ |
| 5 | Đánh giá & báo cáo | COCO AP / PCK trên val2017, FPS, độ trễ | ⏳ |

## 2. Kiến trúc tổng thể

```
Trình duyệt (web/)                              Server (server/)
┌──────────────────────────────┐   JPEG 256px   ┌──────────────────────────────┐
│ Webcam (getUserMedia)        │ ─────────────► │ FastAPI  /ws/pose            │
│ Vẽ skeleton lên video        │                │  ├ Tracker: crop quanh người │
│ Game Three.js (chim, gate)   │ ◄───────────── │  ├ Model 1 (PyTorch, GPU)    │
└──────────────────────────────┘ JSON: khớp +   │  ├ Lọc One-Euro (chống rung) │
                                  control       │  └ controls.py (toán)        │
                                                └──────────────────────────────┘
```

## 3. Model 1 — Human Pose Estimation (phần cốt lõi)

### 3.1 Dataset: COCO-Keypoints 2017
- ~150k người được gán 17 khớp: mũi, mắt, tai, vai, khuỷu, cổ tay, hông, gối, cổ chân.
- Train: ảnh `train2017` có người (tải lẻ từng ảnh, cắt người rồi xoá ảnh gốc để tiết kiệm ổ đĩa).
- Validation: `val2017` (5k ảnh) — đánh giá bằng COCO OKS-AP chuẩn.
- Lọc: người có ≥ 5 khớp được gán, bbox đủ lớn.

### 3.2 Tiền xử lý (`training/prepare_data.py`)
- Cắt vùng vuông quanh mỗi người (1.5× bbox) → lưu JPEG 320×320 + toạ độ khớp đã quy đổi.

### 3.3 Mô hình: SimpleBaseline (Xiao et al., 2018)
- Backbone ResNet (khởi tạo ImageNet — transfer learning) → 3 lớp Deconv → 17 heatmap 64×64.
- Đầu vào 256×256 (vuông, vì tay dang ngang làm vùng thân trên rộng).
- Loss: MSE trên heatmap Gaussian (σ=2), có trọng số theo visibility.
- Giải mã: argmax + dịch 1/4 pixel theo gradient + flip-test khi đánh giá.

### 3.4 Augmentation (quan trọng cho webcam)
- Scale ngẫu nhiên, xoay ±30°, lật ngang (đổi trái↔phải), color jitter.
- **Half-body**: thường xuyên chỉ cắt nửa người trên → giống khung hình webcam thực tế.
- Che ngẫu nhiên (random erasing) để chịu được tay ra khỏi khung.

### 3.5 Huấn luyện
- AdamW, lr 1e-3, cosine decay, AMP (fp16) cho GPU 4GB, batch 32.
- Lưu checkpoint tốt nhất theo AP trên val.

### 3.6 Đánh giá
- COCO Keypoint AP / AP50 / AP75 (pycocotools, dùng GT bbox).
- PCKh / PCK@0.2 riêng cho 8 khớp thân trên (vai, khuỷu, cổ tay, hông) — khớp dùng để điều khiển.
- Tốc độ: ms/khung trên GPU.

## 4. API server (`server/`)
- `POST /predict` — 1 ảnh → JSON khớp (demo qua Swagger `/docs`).
- `WS /ws/pose` — luồng khung hình → khớp + lệnh điều khiển, thời gian thực.
- `GET /health`, `GET /model/info`.
- Tracker: dùng khớp khung trước để xác định vùng cắt khung sau (mất dấu → dùng cả khung hình).
- Lọc One-Euro cho từng khớp để chim không rung.

## 5. Điều khiển bằng toán (`server/controls.py`)
Chuẩn hoá theo độ rộng vai `S` (không phụ thuộc đứng xa/gần):

| Lệnh | Công thức |
|---|---|
| `bank` (rẽ) | góc nghiêng đường nối 2 cổ tay (fallback: khuỷu) so với phương ngang, ∈ [-1, 1] |
| `flap` (vỗ cánh) | vận tốc đi xuống của cổ tay so với vai (đơn vị S/giây), chỉ tính nhịp đập xuống |
| `dive` (bổ nhào) | tay khép: khoảng cách ngang cổ tay–thân nhỏ so với S |
| `spread` | độ dang tay = khoảng cách 2 cổ tay / S (dùng hiển thị & hiệu chỉnh) |

## 6. Web game (`web/`)
- Màn hình chia đôi như video: trên = webcam + skeleton, dưới = game.
- Three.js: địa hình low-poly (noise), cây instanced, bầu trời đêm, mây, chim procedural có cánh đập.
- Vòng cổng (gate) theo đường bay, mũi tên chỉ hướng + khoảng cách (m), thông báo "Gate N!".
- Vật lý: chim luôn bay tới, `bank` → lăn & rẽ, `flap` → lực nâng, `dive` → chúc xuống tăng tốc, trọng lực.
- Bàn phím dự phòng (←/→, Space, ↓) để test khi không có webcam.

## 7. Cấu trúc thư mục
```
PatternRecognition/
├── PLAN.md
├── data/coco/            # dataset (không commit)
├── training/             # prepare_data, dataset, model, train, evaluate
├── checkpoints/          # model đã train
├── server/               # FastAPI + inference + controls
├── web/                  # game Three.js
└── report/               # biểu đồ, kết quả đánh giá
```

## 8. Thứ tự thực hiện
1. Tải dataset (annotations + val2017 + ảnh train có người) ← song song với viết code
2. Viết pipeline dữ liệu + model + train loop, chạy thử nhanh (sanity check)
3. Train đầy đủ trên GPU (vài giờ)
4. Viết server + controls, test với ảnh
5. Viết web game, test bằng bàn phím
6. Ghép webcam → server → game
7. Đánh giá, xuất số liệu & biểu đồ cho báo cáo
