# Mục B — Feature engineering (17 khớp → đặc trưng điều khiển)

Code cho các việc trong `report/KE_HOACH_CONG_VIEC.md` mục B. **Cần bạn quay dữ liệu webcam thật** (bước 1);
mọi bước sau chạy tự động trên dữ liệu đó.

## Cài đặt
```bash
.venv/bin/pip install -r requirements.txt     # đã thêm scikit-learn, pandas
```

## 1. Thu dữ liệu webcam (P0) — việc duy nhất phải làm tay
Cần checkpoint `best.pt` (từ Kaggle run) ở `checkpoints/resnet18/best.pt` hoặc `SKYPOSE_CKPT=...`.

```bash
# 3 session ở 1m (đổi áo/nền/ánh sáng giữa các lần), mỗi session ~3 phút
.venv/bin/python scripts/record_session.py --person ban --distance 1m --rounds 4 --notes "ao den, den bat"
.venv/bin/python scripts/record_session.py --person ban --distance 1m --rounds 4 --seed 1
.venv/bin/python scripts/record_session.py --person ban --distance 1m --rounds 4 --seed 2
# 2 session ở 2m (cho thí nghiệm train 1m → test 2m)
.venv/bin/python scripts/record_session.py --person ban --distance 2m --rounds 4 --seed 3
.venv/bin/python scripts/record_session.py --person ban --distance 2m --rounds 4 --seed 4
```
- Màn hình nhắc từng động tác; 2s "CHUAN BI" trước mỗi đoạn được gán nhãn -1 (bỏ qua).
- Nhãn: `idle` = **dang ngang 2 tay, đứng yên** (tay buông thõng sát người = `dive` theo thiết kế controls.py).
  `left` = hạ tay **trái** thấp; `right` = hạ tay **phải** thấp; `flap` = vẫy mạnh xuống nhiều lần; `dive` = khép tay.
- Script giữ đúng pipeline của game: cắt 3:4, 480px, JPEG q=0.8, không lật gương, One-Euro. Thêm `--landscape` nếu bạn chơi game với `?cam=ngang`.
- Dùng `--mode manual` nếu muốn tự bấm phím 1–5 để gán nhãn.
- File lưu ở `data/sessions/*.npz` (đã bỏ khỏi `.gitignore` để commit được, ~1 MB/session).

## 2. Chạy phân tích
```bash
.venv/bin/python -m analysis.run_all --data data/sessions        # chạy tất cả (~vài phút)
# hoặc từng bước:  python -m analysis.eval_controls --data data/sessions
```
Kết quả (json + png + md) ghi vào `report/feature_eval/`.

| Việc trong kế hoạch | Module | Đầu ra |
|---|---|---|
| P0 Đánh giá bộ điều khiển công thức: P/R từng lệnh, độ trễ flap, báo nhầm idle, chọn ngưỡng bằng lưới/ROC, kiểm leave-one-session-out | `eval_controls` | `controls_eval.*` |
| P0 Chuẩn hoá theo S vs pixel thô (train 1m → test 2m) | `eval_normalization` | `normalization.*` |
| P1 LogReg/SVM/RF/MLP vs công thức (cửa sổ 0.5s) | `compare_ml` | `ml_vs_formula.*` |
| P1 Feature selection (ablation + permutation) | `feature_selection` | `feature_importance.*` |
| P1 One-Euro vs EMA vs Kalman vs không lọc | `compare_filters` | `filters_tradeoff.*` |
| P2 PCA / t-SNE | `embed_features` | `embedding.*` |

## 3. Cách đo (để viết vào báo cáo)
- **Cửa sổ** nhân quả 0.5s (chỉ dùng quá khứ, như realtime). Chỉ chấm các cửa sổ "sạch" (cùng 1 nhãn).
- **Quy tắc quyết định công thức**: flap > dive > rẽ > idle. Ngưỡng mặc định ("đoán") suy ra từ hằng số trong `controls.py`.
- **Chọn ngưỡng**: lưới + F1 một-đấu-tất-cả, lấy *tâm bình nguyên* (không lấy mép lưới). **Leave-one-session-out**: ngưỡng học trên
  các session khác, chấm trên session bị giữ ra → không rò rỉ. Script cảnh báo nếu ngưỡng chọn nằm sát mép lưới.
- **Độ trễ flap**: (a) từ lúc nhãn bắt đầu (gồm phản xạ người), (b) từ lúc tay *thực sự* đập xuống (từ khớp thô, vận tốc > 1 S/s).
- **Độ trễ/jitter của bộ lọc**: jitter = std hiệu liên tiếp/√2 lúc idle; trễ = độ dịch τ làm MSE giữa tín hiệu lọc và tín hiệu thô nhỏ nhất.
- Đã kiểm thử: `python tests/test_analysis.py` — đảm bảo bản vector hoá **trùng khớp** `FlightControls`, đặc trưng bất biến tỉ lệ, OneEuro trùng server.

## 4. Áp ngưỡng tìm được vào game
`controls_eval.json → final_rule` cho `roll_thr_deg, flap_v0, flap_spread, dive_spread, dive_below`. Ánh xạ sang `server/controls.py`:
`flap_v0` ↔ `(v - 1.0)`, `flap_spread` ↔ `spread > 1.8`, `bank_full_deg` ↔ góc rẽ tối đa (ngưỡng rẽ = `roll_thr_deg`).

## 5. Lưu ý trung thực
- `--synthetic` sinh dữ liệu giả **chỉ để chạy thử code**; hình học do mình tự đặt nên mọi phương pháp đều ~100%. **Không đưa vào báo cáo.**
- Dữ liệu chỉ từ 1 người quay → nêu rõ hạn chế này (không khái quát hoá cho người khác).
- Nếu công thức tay ngang/thắng học máy, vẫn là kết quả hợp lệ (nhẹ hơn, giải thích được).
