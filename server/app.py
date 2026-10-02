"""SkyPose API: phục vụ Model 1 (pose estimation tự train) + web game.

Chạy:  .venv/bin/uvicorn server.app:app --port 8765   (model: $SKYPOSE_CKPT, mặc định checkpoints/resnet18/best.pt)
  - Game:     http://localhost:8765/
  - Swagger:  http://localhost:8765/docs
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .controls import FlightControls
from .pose_engine import KEYPOINTS, PoseEngine, PoseSession

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("SKYPOSE_CKPT", ROOT / "checkpoints" / "resnet18" / "best.pt"))
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

app = FastAPI(title="SkyPose API",
              description="Ước lượng tư thế người (model tự train trên COCO-Keypoints) "
                          "và chuyển thành lệnh điều khiển chim bay.")
engine = PoseEngine(CKPT)


def _decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không đọc được ảnh")
    return img


def _kp_json(kp, h, w):
    """Toạ độ chuẩn hoá [0,1] theo kích thước ảnh."""
    return [{"name": n, "x": round(float(kp[i, 0]) / w, 4), "y": round(float(kp[i, 1]) / h, 4),
             "score": round(float(kp[i, 2]), 3)} for i, n in enumerate(KEYPOINTS)]


@app.get("/api/health")
def health():
    return {"status": "ok", "device": engine.device}


@app.get("/api/model/info")
def model_info():
    return engine.info


@app.post("/api/predict", summary="Ảnh -> 17 khớp (JSON)")
async def predict(file: UploadFile = File(...)):
    img = _decode(await file.read())
    h, w = img.shape[:2]
    t = time.perf_counter()
    kp, _ = PoseSession(engine).process(img)
    ctrl = FlightControls()(kp, t)
    return {"width": w, "height": h, "keypoints": _kp_json(kp, h, w), "control": ctrl,
            "latency_ms": round((time.perf_counter() - t) * 1000, 1)}


@app.post("/api/predict/image", summary="Ảnh -> ảnh có vẽ skeleton (JPEG)",
          response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
async def predict_image(file: UploadFile = File(...), min_score: float = 0.3):
    img = _decode(await file.read())
    kp, _ = PoseSession(engine).process(img)
    r = max(2, int(max(img.shape[:2]) / 160))
    for a, b in SKELETON:
        if kp[a, 2] > min_score and kp[b, 2] > min_score:
            cv2.line(img, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (235, 220, 80), r, cv2.LINE_AA)
    for j in range(17):
        if kp[j, 2] > min_score:
            cv2.circle(img, tuple(map(int, kp[j, :2])), r + 2, (40, 200, 255), -1, cv2.LINE_AA)
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


@app.websocket("/ws/pose")
async def ws_pose(ws: WebSocket):
    """Client gửi JPEG (binary) -> server trả JSON {keypoints, control, box, latency_ms}.

    Client nên gửi khung tiếp theo sau khi nhận kết quả (stop-and-wait) để không dồn hàng đợi."""
    await ws.accept()
    session, controls = PoseSession(engine), FlightControls()
    try:
        while True:
            data = await ws.receive_bytes()
            t = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            kp, box = session.process(img, t)
            ctrl = controls(kp, t)
            await ws.send_text(json.dumps({
                "keypoints": [[round(float(kp[i, 0]) / w, 4), round(float(kp[i, 1]) / h, 4),
                               round(float(kp[i, 2]), 3)] for i in range(17)],
                "box": [box[0] / w, box[1] / h, box[2] / w, box[2] / h],
                "control": ctrl,
                "latency_ms": round((time.perf_counter() - t) * 1000, 1),
            }))
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
