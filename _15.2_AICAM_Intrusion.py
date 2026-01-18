#!/usr/bin/env python3
"""
INTRUSIONCAM_PICAMERA2_HARDENED (YOLOv5 ONNX)
--------------------------------------------
Threshold to cross defaults to center view
Change at first lines of main()

Canonical IntrusionCam for this project.

Design goals:
- Raspberry Pi OS + CSI camera only (Picamera2 / libcamera)
- YOLOv5 ONNX via onnxruntime (CPU)
- Evidence-first (preroll + postroll + JSONL events)
- HARD-CODED storage paths (matches other project modules)
- Minimal flags, minimal ambiguity, deterministic behavior
- Detects a human crossing the threshold

This file intentionally avoids:
- Ultralytics
- /dev/video*
- dynamic path guessing
"""

import json
import time
import signal
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2


# ===================== HARD-CODED PATHS =====================
# Adjust here ONCE for this system / deployment

STORAGE_ROOT = Path("/media/user/disk")
MODEL_PATH   = STORAGE_ROOT / "models/yolov5n.onnx"
VIDEO_DIR    = STORAGE_ROOT / "videos/intrusion"
LOG_DIR      = STORAGE_ROOT / "logs/intrusioncam"
EVENTS_PATH  = LOG_DIR / "events.jsonl"

# ============================================================


# ===================== CAMERA CONFIG =====================

CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30

# ==========================================================


# ===================== DETECTION CONFIG ===================

CONF_TH = 0.35
IOU_TH  = 0.45

PREROLL_SEC  = 3.0
POSTROLL_SEC = 4.0
COOLDOWN_SEC = 10.0

# ==========================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def side_of_line(pt, a, b):
    return (pt[0] - a[0]) * (b[1] - a[1]) - (pt[1] - a[1]) * (b[0] - a[0])


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ===================== YOLOv5 ONNX =========================

class YOLOv5ONNX:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h = int(inp.shape[2])
        self.input_w = int(inp.shape[3])
        self.input_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.input_w / w, self.input_h / h)
        nw, nh = int(w * r), int(h * r)
        resized = cv2.resize(img, (nw, nh))
        pad_w = self.input_w - nw
        pad_h = self.input_h - nh
        top = pad_h // 2
        left = pad_w // 2
        out = cv2.copyMakeBorder(
            resized,
            top, pad_h - top,
            left, pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        return out, r, left, top

    def detect(self, frame):
        img, r, dx, dy = self.letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.input_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        preds = self.session.run(None, {self.input_name: img})[0][0]

        boxes = []
        scores = []

        for p in preds:
            obj = p[4]
            cls_conf = p[5]  # class 0 (person)
            conf = obj * cls_conf
            if conf < CONF_TH:
                continue

            cx, cy, w, h = p[:4]
            x1 = (cx - w / 2 - dx) / r
            y1 = (cy - h / 2 - dy) / r
            x2 = (cx + w / 2 - dx) / r
            y2 = (cy + h / 2 - dy) / r

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_TH, IOU_TH)
        out = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                out.append(boxes[i])
        return out


# ===================== MAIN ================================

def main():
    # ---- Boundary (hard-coded on purpose) ----
    LINE_A = (0, CAM_HEIGHT // 2)
    LINE_B = (CAM_WIDTH, CAM_HEIGHT // 2)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print("FATAL: model not found:", MODEL_PATH)
        return 2

    model_hash = sha256_file(MODEL_PATH)
    detector = YOLOv5ONNX(MODEL_PATH)

    events_fp = EVENTS_PATH.open("a", encoding="utf-8")

    # ---- Picamera2 ----
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": float(CAM_FPS)}
    )
    picam.configure(cfg)
    picam.start()

    preroll = deque(maxlen=int(PREROLL_SEC * CAM_FPS))
    last_side = None
    cooldown_until = 0.0

    def handle_exit(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # ---- Startup event ----
    events_fp.write(json.dumps({
        "ts": utc_now(),
        "type": "startup",
        "camera": "picamera2",
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "line": {"a": LINE_A, "b": LINE_B}
    }) + "\n")
    events_fp.flush()

    print("INTRUSIONCAM HARDENED started")

    try:
        while True:
            try:
                rgb = picam.capture_array("main")
            except Exception:
                events_fp.write(json.dumps({
                    "ts": utc_now(),
                    "type": "health",
                    "state": "read_failed"
                }) + "\n")
                break

            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            preroll.append(frame.copy())

            boxes = detector.detect(frame)

            for x1, y1, x2, y2 in boxes:
                cx, cy = bbox_center(x1, y1, x2, y2)
                side = side_of_line((cx, cy), LINE_A, LINE_B)

                if last_side is not None and side * last_side < 0 and time.monotonic() > cooldown_until:
                    clip_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    clip_dir = VIDEO_DIR / clip_id
                    clip_dir.mkdir()

                    clip_path = clip_dir / f"{clip_id}.mp4"
                    h, w = frame.shape[:2]
                    vw = cv2.VideoWriter(
                        str(clip_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        CAM_FPS,
                        (w, h)
                    )

                    for f in preroll:
                        vw.write(f)

                    end = time.monotonic() + POSTROLL_SEC
                    while time.monotonic() < end:
                        try:
                            rgb2 = picam.capture_array("main")
                        except Exception:
                            break
                        vw.write(cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))

                    vw.release()
                    cooldown_until = time.monotonic() + COOLDOWN_SEC

                    events_fp.write(json.dumps({
                        "ts": utc_now(),
                        "type": "event",
                        "event_type": "boundary_crossing",
                        "clip_id": clip_id,
                        "clip_path": str(clip_path),
                        "model_sha256": model_hash
                    }) + "\n")
                    events_fp.flush()

                last_side = side

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        events_fp.write(json.dumps({"ts": utc_now(), "type": "shutdown"}) + "\n")
        events_fp.close()
        print("INTRUSIONCAM stopped")

    return 0


if __name__ == "__main__":
    main()
