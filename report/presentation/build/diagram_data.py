#!/usr/bin/env python3
"""Sinh script window.DIAGRAM_DATA (sơ đồ mermaid) cho presentation SkyPose."""
import json
import sys

SECTIONS = {
    "overview": [
        {"key": "architecture", "label": "🏗️ Kiến trúc tổng quan"},
        {"key": "deployment", "label": "🚀 Môi trường chạy"},
    ],
    "data": [{"key": "data-pipeline", "label": "🗂️ Pipeline dữ liệu"}],
    "model": [{"key": "model-arch", "label": "🧠 Kiến trúc SimpleBaseline"}],
    "training": [
        {"key": "train-loop", "label": "🏋️ Vòng lặp huấn luyện"},
        {"key": "kaggle-pipeline", "label": "☁️ Pipeline Kaggle"},
    ],
    "results": [{"key": "eval-protocol", "label": "📏 Giao thức đánh giá"}],
    "inference": [
        {"key": "inference-pipeline", "label": "🎯 Suy luận 1 khung"},
        {"key": "tracker-state", "label": "🔁 Máy trạng thái tracker"},
    ],
    "controls": [{"key": "controls-math", "label": "📐 Khớp → lệnh"}],
    "api": [{"key": "ws-sequence", "label": "🌐 Trình tự WebSocket"}],
    "game": [{"key": "game-state", "label": "🎮 Máy trạng thái game"}],
    "flow": [{"key": "e2e", "label": "🔄 End-to-end"}],
    "compare": [
        {"key": "compare-axes", "label": "⚖️ 4 trục so sánh"},
        {"key": "pipeline-chart", "label": "📈 Mean OKS theo chiến lược"},
        {"key": "speed-chart", "label": "⚡ Độ trễ theo backbone"},
    ],
    "future": [{"key": "future-arch", "label": "🚀 Kiến trúc hướng tới"}],
    "roadmap": [{"key": "roadmap", "label": "🗺️ Gantt"}],
}

DIAGRAMS = {}

DIAGRAMS["architecture"] = {
    "title": "Kiến trúc tổng quan SkyPose",
    "mermaid": """flowchart LR
    subgraph OFF["Offline — huấn luyện"]
        COCO[("COCO-Keypoints 2017")] --> PREP["prepare_data.py<br/>crop 320×320"]
        PREP --> TRAIN["train.py<br/>SimpleBaseline-R18"]
        TRAIN --> EVAL["evaluate.py<br/>AP · PCK · latency"]
        TRAIN --> CK[("best.pt")]
    end
    subgraph SRV["Server — FastAPI"]
        WS["/ws/pose"] --> SESS["PoseSession<br/>crop tracker"]
        SESS --> ENG["PoseEngine<br/>FP16 + flip-test"]
        ENG --> EURO["One-Euro ×17"]
        EURO --> CTRL["FlightControls<br/>bank · flap · dive"]
        REST["/api/predict"] --> SESS
    end
    subgraph WEB["Trình duyệt"]
        CAM["Webcam"] --> POSE["pose.js<br/>cắt dọc 3:4 · JPEG 360×480"]
        GAME["main.js<br/>vật lý + Three.js"]
        HUD["Khung xương + HUD"]
    end
    CK --> ENG
    POSE -- "binary JPEG" --> WS
    CTRL -- "JSON khớp + lệnh" --> POSE
    POSE --> GAME
    POSE --> HUD
""",
}

DIAGRAMS["deployment"] = {
    "title": "Môi trường chạy",
    "mermaid": """flowchart TB
    subgraph K["Kaggle — 2× Tesla T4"]
        NB["skypose_train_kaggle.ipynb"] --> CL["git clone repo"]
        CL --> PY["training/*.py"]
        PY --> ZIP["skypose_…_kaggle.zip<br/>log · csv · figures · best.pt"]
    end
    subgraph L["Laptop — RTX A2000 4 GB"]
        IMP["scripts/import_kaggle_run.py"] --> RUNS[("runs/&lt;run&gt;/")]
        RUNS --> UV["uvicorn server.app:app :8765"]
        RUNS --> NBT["notebooks/test_pose_images.ipynb"]
        UV --> BR["Trình duyệt localhost:8765"]
    end
    GH[("GitHub repo")] --> CL
    ZIP -- "tải về" --> IMP
""",
}

DIAGRAMS["data-pipeline"] = {
    "title": "Pipeline dữ liệu",
    "mermaid": """flowchart LR
    A[("person_keypoints_*.json")] --> B{"Lọc người<br/>không crowd, ≥ 5 khớp,<br/>bbox đủ lớn"}
    B --> C["Tải lẻ ảnh<br/>images.cocodataset.org"]
    C --> D["Crop vuông 1,5× bbox<br/>→ 320×320 JPEG"]
    D --> E[("crops/{split}/{ann_id}.jpg")]
    D --> F[("{split}_index.npz<br/>keypoints · center · side · area")]
    E --> G["CocoPoseCrops<br/>__getitem__"]
    F --> G
    G --> H["Augment<br/>half-body · scale · xoay · lật · màu · erasing"]
    H --> I["input uint8 3×256×256"]
    H --> J["heatmap 17×64×64 + weight"]
""",
}

DIAGRAMS["model-arch"] = {
    "title": "SimpleBaseline — ResNet18 + 3 deconv",
    "mermaid": """flowchart LR
    X["Ảnh RGB<br/>3×256×256"] --> C1["conv1 + maxpool<br/>64×64×64"]
    C1 --> L1["layer1<br/>64×64×64"]
    L1 --> L2["layer2<br/>128×32×32"]
    L2 --> L3["layer3<br/>256×16×16"]
    L3 --> L4["layer4<br/>512×8×8"]
    L4 --> D1["Deconv 4×4 s2<br/>256×16×16"]
    D1 --> D2["Deconv 4×4 s2<br/>256×32×32"]
    D2 --> D3["Deconv 4×4 s2<br/>256×64×64"]
    D3 --> H["Conv 1×1<br/>17×64×64"]
    H --> DEC["argmax + ¼ px<br/>→ (x, y, conf) ×17"]
    style L4 fill:#e0f2fe
    style H fill:#fef3c7
""",
}

DIAGRAMS["train-loop"] = {
    "title": "Vòng lặp huấn luyện (train.py)",
    "mermaid": """flowchart TB
    S["Khởi tạo: PoseNet(ImageNet), AdamW, warm-up + cosine, GradScaler"] --> R{"--resume?"}
    R -- có --> LD["Nạp last.pt: model, opt, sched, scaler, epoch, best"]
    R -- không --> EP
    LD --> EP["Epoch e"]
    EP --> IT["Batch uint8 → normalize_batch (GPU)"]
    IT --> FW["forward FP16 (autocast)"]
    FW --> LS["weighted MSE (FP32)"]
    LS --> BW["scaler.scale(loss).backward → step → sched.step"]
    BW --> IT
    BW --> EV["Hết epoch: predict flip-test trên val"]
    EV --> M["COCO AP + PCK → train_log.csv"]
    M --> SV["Lưu last.pt"]
    SV --> B{"AP > best?"}
    B -- có --> BP["Lưu best.pt"]
    B -- không --> NX
    BP --> NX{"e < epochs?"}
    NX -- có --> EP
    NX -- không --> END["Xong: best AP"]
""",
}

DIAGRAMS["kaggle-pipeline"] = {
    "title": "Pipeline Kaggle (run 20261002-1405)",
    "mermaid": """gantt
    title Thời gian từng bước trên Kaggle 2× T4 (phút)
    dateFormat X
    axisFormat %s
    section Chuẩn bị
    01 setup (clone + pip)      :0, 1
    02 prepare val (0,24)       :1, 2
    03 prepare train (5,4)      :2, 8
    section Huấn luyện
    04 train 20 epoch (320)     :8, 328
    section Đánh giá
    05 evaluate (0,5)           :328, 329
    06 figures + zip            :329, 330
""",
}

DIAGRAMS["eval-protocol"] = {
    "title": "Hai giao thức đánh giá",
    "mermaid": """flowchart TB
    V[("COCO val2017<br/>5.000 ảnh")] --> G1
    V --> G2
    subgraph G1["Giao thức 1 — chuẩn COCO (Chapter 04)"]
        A1["Crop theo bbox ground-truth<br/>6.352 người"] --> A2["Model + flip-test"]
        A2 --> A3["pycocotools OKS → AP/AR"]
        A2 --> A4["PCK@α theo từng khớp<br/>+ 8 khớp điều khiển"]
    end
    subgraph G2["Giao thức 2 — pipeline thật (Chapter 10.3)"]
        B1["500 ảnh 1 người, ≥ 10 khớp<br/>không dùng bbox"] --> B2["Lượt 1: cả ảnh"]
        B2 --> B3["Lượt 2: crop quanh khớp tìm được"]
        B3 --> B4["OKS từng ảnh → mean, % > 0,5, % > 0,75"]
    end
""",
}

DIAGRAMS["inference-pipeline"] = {
    "title": "Suy luận một khung hình (PoseEngine.predict_crop)",
    "mermaid": """flowchart LR
    I["Ảnh BGR"] --> M["square_affine(center, side)"]
    M --> W["warpAffine → 256×256"]
    W --> T["Chuẩn hoá ImageNet → tensor"]
    T --> F["Ghép [ảnh, ảnh lật] → batch 2"]
    F --> N["PoseNet FP16"]
    N --> HF["Lật heatmap ảnh lật,<br/>đổi trái↔phải, dịch 1 px"]
    HF --> AV["Trung bình 2 heatmap"]
    AV --> D["decode: argmax + ¼ px"]
    D --> S["× 4 → toạ độ input 256"]
    S --> INV["Affine ngược → pixel ảnh gốc"]
    INV --> O["(17, 3): x, y, conf"]
""",
}

DIAGRAMS["tracker-state"] = {
    "title": "Máy trạng thái của crop tracker (PoseSession)",
    "mermaid": """stateDiagram-v2
    [*] --> FullFrame
    FullFrame: Crop cả khung hình
    Tracking: Crop quanh người (×1,6, làm mượt 50/50)
    FullFrame --> Tracking: ≥ 4 khớp tin cậy (17 khớp) và thấy 2 vai
    FullFrame --> FullFrame: không đủ khớp
    Tracking --> Tracking: vẫn đủ khớp
    Tracking --> FullFrame: mất dấu
    note right of Tracking
        Mỗi khung: One-Euro cho 17 khớp
        trong toạ độ chuẩn hoá [0,1]
    end note
""",
}

DIAGRAMS["controls-math"] = {
    "title": "Từ khớp đến lệnh điều khiển (FlightControls)",
    "mermaid": """flowchart TB
    K["17 khớp đã lọc"] --> Q{"Thấy 2 vai?"}
    Q -- không --> NP["present = false<br/>flap giảm dần ×0,8"]
    Q -- có --> S["S = |vai trái − vai phải|"]
    S --> TIP["Đầu cánh = cổ tay<br/>(fallback: khuỷu)"]
    TIP --> Q2{"Có cả 2 đầu cánh?"}
    Q2 -- không --> P["present = true, lệnh = 0"]
    Q2 -- có --> B["bank = atan2(Δy, Δx) / 40°"]
    Q2 -- có --> SP["spread = |Δx| / S"]
    Q2 -- có --> EL["elev = độ nâng tay / S"]
    SP --> DV["dive = f(2,2 − spread) · f(tay dưới vai)"]
    EL --> V["v = −Δelev/Δt (8 mẫu)"]
    V --> FL["flap = max(clamp((v−1)/4), 0,85·flap)<br/>chỉ khi spread > 1,8"]
    SP --> FL
""",
}

DIAGRAMS["ws-sequence"] = {
    "title": "Trình tự WebSocket /ws/pose (stop-and-wait)",
    "mermaid": """sequenceDiagram
    autonumber
    participant C as pose.js (trình duyệt)
    participant S as FastAPI /ws/pose
    participant P as PoseSession
    participant E as PoseEngine (GPU)
    participant F as FlightControls
    C->>S: mở WebSocket
    S->>S: tạo PoseSession + FlightControls riêng
    loop mỗi khung hình
        C->>C: video → cắt giữa 3:4 → canvas 360×480 → JPEG q 0,8
        C->>S: gửi bytes
        S->>P: process(img, t)
        P->>E: predict_crop(img, box)
        E-->>P: 17 × (x, y, conf)
        P->>P: cập nhật box, One-Euro
        P-->>S: kp, box
        S->>F: controls(kp, t)
        F-->>S: bank, flap, dive, spread, wing
        S-->>C: JSON {keypoints, box, control, latency_ms}
        C->>C: vẽ khung xương, cập nhật lệnh, đo RTT
    end
    alt socket đóng
        C->>C: hiện "Mất kết nối tới server pose"
    end
""",
}

DIAGRAMS["game-state"] = {
    "title": "Máy trạng thái game (main.js)",
    "mermaid": """stateDiagram-v2
    [*] --> menu
    menu --> calib: Chơi bằng webcam
    menu --> countdown: Chơi bằng bàn phím
    calib --> countdown: spread > 3 giữ 1,2 s
    countdown --> play: hết 3 s
    play --> play: qua cổng N
    play --> Autopilot: mất người > 0,8 s
    Autopilot --> play: thấy lại người
    play --> end: qua đủ 16 cổng
    end --> calib: Chơi lại (webcam)
    end --> countdown: Chơi lại (bàn phím)
""",
}

DIAGRAMS["e2e"] = {
    "title": "End-to-end: dữ liệu → model → cú vỗ cánh",
    "mermaid": """sequenceDiagram
    autonumber
    participant D as COCO
    participant K as Kaggle
    participant R as runs/
    participant U as Người chơi
    participant B as Trình duyệt
    participant S as Server
    D->>K: annotation + ảnh
    K->>K: prepare → train 20 epoch → evaluate
    K->>R: zip (best.pt, log, figures)
    R->>S: SKYPOSE_CKPT=…/best.pt
    U->>B: mở localhost:8765, bật webcam
    B->>S: GET /api/model/info
    S-->>B: resnet18, epoch 20, AP 61,8
    U->>B: dang tay (hiệu chỉnh)
    loop ~30 khung/giây
        B->>S: JPEG
        S-->>B: khớp + bank/flap/dive
        B->>B: vật lý bay + render
    end
    U->>B: vẫy tay xuống
    B->>B: flap → chim bay lên, qua cổng
""",
}

DIAGRAMS["compare-axes"] = {
    "title": "Bốn trục so sánh",
    "mermaid": """mindmap
  root((SkyPose<br/>so sánh))
    A Mô hình
      Độ dài train ep3 vs ep20 ✓
      Backbone R18 / R34 / R50
      Heatmap vs SimCC vs hồi quy
      Input 256×256 vs 256×192
    B Suy luận
      Flip-test ✓
      1 lượt vs 2 lượt ✓
      Crop 13 vs 17 khớp ✓
      FP32 / FP16 / CPU ✓
    C Hệ thống
      Tự train vs MediaPipe / MoveNet / RTMPose
      Server vs trong trình duyệt
    D Điều khiển
      Hình học vs phân loại cử chỉ
      Tham số One-Euro
""",
}

DIAGRAMS["pipeline-chart"] = {
    "title": "Mean OKS trên 500 ảnh val2017 (không có GT bbox, flip bật)",
    "mermaid": """xychart-beta
    title "Mean OKS theo chiến lược cắt (flip-test bật)"
    x-axis ["1 lượt", "2 lượt · 13 khớp", "2 lượt · 17 khớp"]
    y-axis "Mean OKS" 0.4 --> 0.7
    bar [0.547, 0.591, 0.617]
""",
}

DIAGRAMS["speed-chart"] = {
    "title": "Độ trễ forward theo backbone (RTX A2000 Laptop, bs 1)",
    "mermaid": """xychart-beta
    title "ms / ảnh — cột nhạt: FP32 · cột đậm: FP16"
    x-axis ["ResNet18", "ResNet34", "ResNet50"]
    y-axis "ms" 0 --> 9
    bar [3.57, 5.69, 7.78]
    bar [2.26, 3.67, 5.00]
""",
}

DIAGRAMS["future-arch"] = {
    "title": "Kiến trúc hướng tới: model chạy trong trình duyệt",
    "mermaid": """flowchart LR
    subgraph TRAIN["Huấn luyện"]
        T50["Teacher R50 / HRNet"] -- distillation --> S18["Student R18<br/>+ DARK / SimCC"]
        WC[("Khung webcam tự thu")] -- fine-tune --> S18
        S18 --> GATE{"Cổng đánh giá<br/>AP ≥ ngưỡng?"}
    end
    GATE -- đạt --> ONNX["Export ONNX<br/>+ INT8"]
    subgraph BROWSER["Trình duyệt người chơi"]
        CAM["Webcam"] --> ORT["ONNX Runtime Web<br/>WebGPU / WASM"]
        ORT --> TRK["Tracker + One-Euro (JS)"]
        TRK --> CT["controls (JS)"]
        CT --> G["Game Three.js"]
    end
    ONNX -- "hosting tĩnh / CDN" --> ORT
    G -. "điểm số (tuỳ chọn)" .-> LB[("Leaderboard API")]
""",
}

DIAGRAMS["roadmap"] = {
    "title": "Lộ trình (mốc đã làm theo git/log; mốc tiếp theo là dự kiến)",
    "mermaid": """gantt
    title Lộ trình SkyPose
    dateFormat YYYY-MM-DD
    section Đã làm
    Dữ liệu, model, server, game    :done, 2026-10-02, 1d
    Train laptop 3 epoch            :done, 2026-10-02, 1d
    Train Kaggle 20 epoch           :done, 2026-10-02, 2d
    Sửa lỗi WebSocket               :done, 2026-10-12, 1d
    Notebook test + so sánh         :done, 2026-10-13, 1d
    section Dự kiến
    P0 DARK, đổi checkpoint, pytest :active, 2026-10-14, 3d
    3 báo cáo                       :2026-10-14, 7d
    P1 train dài, R50, distill      :2026-10-17, 10d
    P1 ONNX Runtime Web             :2026-10-24, 7d
    P2 game, nhiều người            :2026-10-31, 10d
""",
}


def main() -> None:
    missing = [d["key"] for v in SECTIONS.values() for d in v if d["key"] not in DIAGRAMS]
    if missing:
        print("Thiếu định nghĩa diagram: %s" % ", ".join(sorted(set(missing))), file=sys.stderr)
        raise SystemExit(1)
    payload = json.dumps({"sections": SECTIONS, "diagrams": DIAGRAMS}, ensure_ascii=False, indent=4)
    sys.stdout.write("""    <script>
/* Mermaid diagram data cho presentation SkyPose — sinh tự động bởi build/diagram_data.py. */
(function () {
    window.DIAGRAM_DATA = %s;
})();
    </script>
""" % payload)


if __name__ == "__main__":
    main()
