# SkyPose — Tài liệu công việc cần làm (đồ án Phân tích & Nhận dạng mẫu)

> Mục đích: biến sản phẩm hiện có (model pose tự train + API + game chim bay) thành **một đồ án môn học đúng nghĩa**:
> có **nghiên cứu** (giả thuyết → thí nghiệm → kết luận), có **quá trình phát triển** (thử → thất bại → cải tiến),
> có **câu chuyện để kể**, và làm rõ **thuật toán / tiền xử lý / feature engineering** — là những thứ giảng viên chấm.

Cập nhật: 2026-11-13. Tick `[x]` = đã xong (có bằng chứng trong repo), `[ ]` = cần làm.

---

## 0. Đánh giá hiện trạng (trung thực)

**Đã có (mạnh):**
- Model SimpleBaseline ResNet18 tự train trên COCO-Keypoints, 20 epoch trên Kaggle: **AP 61.8, AP50 86.3, PCK@0.1 (khớp điều khiển) 88%** (`runs/20261002-1405_resnet18_kaggle/eval_resnet18.json`).
- Pipeline dữ liệu: crop quanh người, half-body augmentation, flip đổi trái↔phải, random erasing, màu trên uint8.
- Suy luận realtime: tracker crop, flip-test, One-Euro filter, FastAPI REST + WebSocket.
- Điều khiển bằng hình học (`server/controls.py`): bank, flap, dive, spread chuẩn hoá theo độ rộng vai.
- Game Three.js; slide HTML (`report/presentation/`); có thí nghiệm đầu tiên (`experiments/results_20261013.json`).

**Còn thiếu so với yêu cầu "đồ án nghiên cứu":**
1. Chưa có **bảng ablation** chứng minh từng quyết định tiền xử lý/augmentation có ích (hiện chỉ *nói* là có ích).
2. **Feature engineering** mới là công thức tay, chưa được **đánh giá định lượng** (chưa có ground truth cho lệnh điều khiển, chưa so với phương pháp học).
3. Chưa có **câu chuyện liền mạch** (vấn đề → thất bại → phát hiện → cải tiến → kết quả) trong một văn bản báo cáo.
4. 3 báo cáo cuối (tổng quan / nhật ký train / đánh giá hiệu năng) chưa viết.
5. Chưa đo **domain gap COCO → webcam thật** — đây chính là "điểm chuyện" hay nhất của đề tài.

---

## 1. Câu chuyện xuyên suốt (storyline)

Khung kể 6 chương, mỗi chương = một câu hỏi + một thứ đã làm + một con số:

| Chương | Câu hỏi | Nội dung / bằng chứng |
|---|---|---|
| 1. Vấn đề | Làm sao biến cơ thể thành tay lái, không cần cảm biến? | Game *skypeck*; yêu cầu: realtime, 1 webcam, người đứng xa/gần khác nhau |
| 2. Dữ liệu | Dữ liệu thật trông thế nào, tại sao không dùng thẳng? | COCO: ảnh nhiều người, nhiều khớp bị che, toàn thân; webcam: nửa người trên, 1 người → **domain gap** |
| 3. Tiền xử lý | Biến dữ liệu COCO thành dữ liệu giống webcam ra sao? | Crop 1.5×bbox, half-body, affine, flip-swap, erasing (+ ablation) |
| 4. Mô hình | Vì sao heatmap + ResNet18? | Tự train, laptop 4GB thất bại/nóng → chuyển Kaggle; AP theo epoch (3→20: 48.3→61.8) |
| 5. Feature engineering | Từ 17 điểm → "rẽ/vỗ/bổ nhào" thế nào? | Đặc trưng hình học chuẩn hoá theo vai, vận tốc, lọc; so với classifier học |
| 6. Hệ thống & bài học | Chạy thật thì sao? | FPS, độ trễ, lỗi gặp (jitter, mất khớp, tay ra khỏi khung), cách vá |

**Các "tình tiết" thật đã xảy ra, nên đưa vào câu chuyện** (đây là phần ăn điểm "quá trình phát triển"):
- Laptop A2000 tụt xuống 20W khi train lâu → ngừng ở epoch 3 → chuyển Kaggle T4 (xem `memory/gpu-thermal-throttle`).
- Tối ưu dataloader cho máy ít CPU: augmentation màu bằng OpenCV uint8 (~6× nhanh), chuẩn hoá trên GPU (uint8 nhẹ hơn float32 4×).
- Lần đầu AP 48.3 ở epoch 3 → 61.8 ở epoch 20; flip-test +1.5 AP (60.3 → 61.8).
- Pipeline "2 pass" (crop lại theo kết quả pass 1) tăng mean OKS 0.547 → 0.617 (`results_20261013.json`).
- Đổi tên/cấu hình chạy Kaggle, script import run (`scripts/import_kaggle_run.py`) để lưu log đầy đủ.

---

## 2. Danh sách công việc

Ưu tiên: **P0** = bắt buộc để đủ đồ án; **P1** = nên có để điểm cao; **P2** = nếu còn thời gian.

### A. Tiền xử lý dữ liệu (Preprocessing) — cần viết rõ + chứng minh

Mô tả từng bước (đầu vào → phép biến đổi → đầu ra → *vì sao*), đã có trong code, cần đưa vào báo cáo:

| Bước | Code | Lý do (nhận dạng mẫu) |
|---|---|---|
| Lọc người ≥5 khớp, bbox đủ lớn | `training/prepare_data.py` | Loại mẫu nhiễu / nhãn không đủ thông tin |
| Crop vuông 1.5×bbox → 320px | `prepare_data.py` | Chuẩn hoá vị trí & tỉ lệ, giảm dung lượng |
| Affine (scale/xoay ±30°/dịch) | `pose_utils.square_affine` | Bất biến với tư thế & khoảng cách |
| Lật ngang + hoán đổi trái↔phải | `FLIP_INDEX` | Tăng gấp đôi dữ liệu mà không sai nhãn |
| Half-body (40%) | `dataset._augment_box` | Mô phỏng khung webcam |
| Contrast/brightness/saturation | `dataset._color` | Bất biến ánh sáng |
| Random erasing | `dataset._color` | Chịu che khuất, tay ra khỏi khung |
| Chuẩn hoá ImageNet | `normalize_batch` | Khớp với trọng số tiền huấn luyện |
| Heatmap Gaussian σ=2, trọng số visibility | `make_heatmaps` | Biểu diễn nhãn mượt, bỏ khớp không gán nhãn khỏi loss |

- [ ] **P0** Vẽ sơ đồ pipeline tiền xử lý + **hình minh hoạ trước/sau** (ảnh gốc → crop → augment → heatmap) cho 4–6 mẫu.
- [ ] **P0** **Ablation augmentation** (mỗi cấu hình train ngắn ~5–8 epoch trên Kaggle, cùng seed, đo AP + PCK khớp điều khiển):

  | Cấu hình | Half-body | Flip | Xoay | Erasing | Color |
  |---|---|---|---|---|---|
  | Baseline (không aug) | ✗ | ✗ | ✗ | ✗ | ✗ |
  | + flip/scale/xoay | ✗ | ✓ | ✓ | ✗ | ✗ |
  | + color + erasing | ✗ | ✓ | ✓ | ✓ | ✓ |
  | Đầy đủ (đang dùng) | ✓ | ✓ | ✓ | ✓ | ✓ |

  Chỉ số quan trọng: đánh giá trên **tập mô phỏng webcam** (cắt half-body của val2017) — dự đoán half-body thắng rõ ở đây dù AP COCO thường gần bằng nhau. Đó là kết luận đáng kể.
- [ ] **P1** Ablation σ heatmap (1.5 / 2 / 3) và kích thước input (192 / 256) — đánh đổi AP ↔ FPS.
- [ ] **P2** Thống kê dữ liệu (EDA): phân bố số khớp được gán, kích thước bbox, tỉ lệ visibility theo từng khớp → biện luận vì sao lọc ≥5 khớp.

### B. Feature engineering (từ 17 khớp → đặc trưng điều khiển) — điểm nhấn học thuật

Hiện tại đặc trưng là công thức tay. Cần: (1) **liệt kê và giải thích** từng đặc trưng, (2) **đo chất lượng**, (3) **so sánh với cách học**.

Bảng đặc trưng cần viết trong báo cáo:

| Đặc trưng | Công thức | Ý nghĩa | Bất biến với |
|---|---|---|---|
| Độ rộng vai `S` | ‖LS − RS‖ | Đơn vị chuẩn | Khoảng cách tới camera |
| `roll` | atan2(Δy, Δx) 2 cổ tay | Rẽ | Tỉ lệ, vị trí |
| `spread` | \|Δx cổ tay\| / S | Dang/khép tay | Tỉ lệ |
| `wing_l/r` | góc vai→cổ tay | Hiển thị cánh | Tỉ lệ |
| `below` | (y cổ tay − y vai)/S | Tay thấp hơn vai | Tỉ lệ |
| `elev`, vận tốc `v` | d(elev)/dt | Vỗ cánh | Tỉ lệ, theo thời gian |
| Fallback khuỷu | khi cổ tay conf < ngưỡng | Chịu mất khớp | — |

- [ ] **P0** **Thu thập tập dữ liệu nhỏ từ webcam** (10–15 phút, bạn tự quay): ghi chuỗi khớp + **nhãn thủ công** theo từng đoạn (trái / phải / vỗ / bổ nhào / đứng yên). Viết script `scripts/record_session.py` (lưu `.npz`: khớp, conf, timestamp, nhãn). Đây là dữ liệu thật duy nhất để đánh giá phần điều khiển.
- [ ] **P0** **Đánh giá bộ điều khiển công thức**: với tập trên, tính độ chính xác/recall từng lệnh, độ trễ phát hiện (ms từ lúc làm động tác tới lúc `flap`>ngưỡng), tỉ lệ báo nhầm khi đứng yên. Chọn ngưỡng (`bank_full`, `v−1.0`, `spread>1.8`) bằng **ROC / lưới ngưỡng** trên tập huấn luyện nhỏ, kiểm trên tập còn lại — thay vì "đoán".
- [ ] **P0** **So sánh đặc trưng chuẩn hoá vs thô**: cùng bộ phân loại, một bên dùng toạ độ pixel thô, một bên dùng đặc trưng chuẩn hoá theo `S` → chứng minh chuẩn hoá giúp khi đổi khoảng cách (quay ở 1m và 2m, train ở 1m, test ở 2m).
- [ ] **P1** **Phương án học máy cho cùng nhiệm vụ**: Logistic Regression / SVM / Random Forest / MLP nhỏ trên vector đặc trưng (kèm cửa sổ trượt 0.5s: trung bình, độ lệch, vận tốc). So sánh với công thức tay: độ chính xác, độ trễ, khả năng giải thích. Kết luận trung thực dù công thức thắng (đó vẫn là kết quả).
- [ ] **P1** **Feature selection**: phân tích độ quan trọng (permutation importance / ablation bỏ từng đặc trưng) → đặc trưng nào thực sự cần.
- [ ] **P1** **Lọc tín hiệu**: so sánh One-Euro vs EMA vs Kalman vs không lọc theo (jitter tại lúc đứng yên ↔ độ trễ khi chuyển động nhanh). Vẽ biểu đồ đánh đổi. Đã có `pipeline_raw` làm nền.
- [ ] **P2** PCA/t-SNE trên vector đặc trưng đã gán nhãn → cho thấy các lớp động tác tách được (hình đẹp cho báo cáo).

### C. Thuật toán / mô hình — phần cần giải thích sâu

Phải trình bày được (kèm công thức) — viết vào báo cáo tổng quan:
- [ ] **P0** Bài toán: hồi quy toạ độ trực tiếp vs heatmap; vì sao heatmap (không gian nhãn mượt, gradient tốt hơn).
- [ ] **P0** Kiến trúc: ResNet18 → 3 deconv → 17 heatmap 64×64; sơ đồ kích thước tensor từng tầng; số tham số.
- [ ] **P0** Loss: MSE có trọng số visibility `L = (1/K) Σ w_k ‖Ĥ_k − H_k‖²`.
- [ ] **P0** Giải mã: argmax + dịch 1/4 pixel; chuyển ngược toạ độ bằng nghịch đảo affine.
- [ ] **P0** Chỉ số: **OKS** `exp(−d²/(2 s² k²))`, AP/AP50/AP75, PCK — giải thích ý nghĩa, vì sao dùng PCK khớp điều khiển.
- [ ] **P0** Transfer learning: so sánh **có / không** khởi tạo ImageNet (train ngắn) → tốc độ hội tụ.
- [ ] **P1** So sánh backbone: ResNet18 vs ResNet50 (AP vs ms/khung vs tham số) — bảng "Pareto" độ chính xác-tốc độ.
- [ ] **P1** Flip-test, pipeline 2-pass, tracker crop: bảng đóng góp từng thành phần (đã có số sơ bộ trong `results_20261013.json`, cần chạy lại sạch + ghi lại cấu hình).
- [ ] **P1** One-Euro filter: công thức `α = 1/(1+τ/Te)`, `fc = fcmin + β|dx̂|`.
- [ ] **P2** Phân tích lỗi (error analysis): 20 ca sai nặng nhất → phân loại nguyên nhân (che khuất, mờ, nhiều người, tay ngoài khung).

### D. Đánh giá thực nghiệm hệ thống

- [ ] **P0** Domain gap: đo PCK trên (a) COCO val, (b) ~100–200 khung webcam tự gán nhãn nhanh cho 6 khớp thân trên (dùng công cụ gán nhãn đơn giản hoặc OKS so với model lớn làm "pseudo-label" — nói rõ trong báo cáo đây là pseudo-label).
- [ ] **P0** FPS & độ trễ end-to-end: webcam → server → trình duyệt (ms), phân rã từng giai đoạn (JPEG, mạng, model, filter, controls).
- [ ] **P1** Test điều kiện: ánh sáng yếu, nền rối, mặc áo cùng màu nền, đứng xa 1m/2m, tay ra khỏi khung.
- [ ] **P1** Người chơi thử (3–5 người): tỉ lệ qua cổng, khảo sát 3 câu hỏi (dễ điều khiển? trễ? mỏi tay?).

### E. Báo cáo & trình bày (3 báo cáo đã cam kết + slide)

- [ ] **P0** Báo cáo 1 — **Tổng quan** (`report/01_tong_quan.md`): theo storyline mục 1; có sơ đồ kiến trúc, bảng tiền xử lý, bảng đặc trưng.
- [ ] **P0** Báo cáo 2 — **Nhật ký huấn luyện** (`report/02_nhat_ky_train.md`): kể theo thời gian — laptop, throttle, chuyển Kaggle, các lần chạy, đường cong loss/AP/lr (có sẵn `runs/.../figures`), bài học.
- [ ] **P0** Báo cáo 3 — **Đánh giá hiệu năng** (`report/03_danh_gia.md`): bảng AP/PCK, ablation, so sánh, FPS, domain gap, error analysis.
- [ ] **P0** Mục **"Hạn chế & hướng phát triển"** trung thực (1 trang).
- [ ] **P1** Cập nhật slide `report/presentation/` bằng số liệu ablation mới (chương 10–11).
- [ ] **P1** Video demo 1–2 phút: webcam + skeleton + game, kèm cảnh model sai và cách xử lý.
- [ ] **P2** README: 1 lệnh để tái lập kết quả (seed, phiên bản thư viện, cấu hình).

---

## 3. Thứ tự thực hiện đề xuất

| Giai đoạn | Việc | Kết quả |
|---|---|---|
| 1 | Ghi dữ liệu webcam + gán nhãn (B-P0 thu thập) | Tập dữ liệu thật để làm mọi thứ sau |
| 2 | Đánh giá điều khiển công thức, chọn ngưỡng, chuẩn hoá vs thô | Phần feature engineering có số |
| 3 | Ablation augmentation trên Kaggle (chạy nền song song) | Bảng chứng minh tiền xử lý |
| 4 | Phương án học máy + lọc tín hiệu + feature importance | So sánh phương pháp |
| 5 | Domain gap, FPS/độ trễ, error analysis | Phần đánh giá |
| 6 | Viết 3 báo cáo, cập nhật slide, quay demo | Nộp |

Lưu ý tài nguyên: laptop dễ throttle khi train dài (ước lượng theo tốc độ bền vững), nên các lần ablation chạy trên **Kaggle**, mỗi cấu hình ngắn; phân tích đặc trưng/classifier chỉ cần CPU.

---

## 4. Tiêu chí tự kiểm trước khi nộp

- [ ] Mỗi quyết định thiết kế đều có **một con số hoặc một hình** bảo vệ.
- [ ] Có ít nhất **một kết quả "ngược dự đoán"** hoặc một thất bại được kể lại và giải thích.
- [ ] Phân biệt rõ phần **tự train/tự làm** và phần dùng sẵn (ImageNet, pycocotools, Three.js, thư viện One-Euro).
- [ ] Mọi số trong báo cáo truy được về file trong `runs/` hoặc `report/presentation/experiments/`.
- [ ] Có phần hạn chế: COCO ≠ webcam, 17 khớp nhưng chỉ dùng 8–9, AP 61.8 thấp hơn model SOTA nhưng đủ cho điều khiển.
