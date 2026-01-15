#!/usr/bin/env python3
# rpiAICAM_picamera2_cv_mobilenetssd.py
# Picamera2 + OpenCV DNN (Caffe MobileNet-SSD) — still capture with:
#   - Professional overlay
#   - "person detected" => save image + write one-line event log
#   - Debug mode outputs (raw + overlay + dnn input)
#   - Hashing/version stamping (model hashes, OpenCV, Python)
#
# Requires (local files):
#   /home/fry/models/MobileNetSSD_deploy.prototxt
#   /home/fry/models/MobileNetSSD_deploy.caffemodel
#
# Output (mount/fallback):
#   image_<ts>.jpg           (overlay)
#   meta_<ts>.json
#   events.log               (append-only, one line per event)
# Debug mode (if DEBUG=True):
#   raw_<ts>.jpg             (no overlay)
#   dnn_<ts>.jpg             (300x300 image fed to DNN)

import os
import json
import hashlib
import platform
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2

# -----------------------------
# Storage
# -----------------------------
MOUNT = "/media/usr/disk"
OUT_SUBDIR = "images"
FALLBACK_DIR = "/home/usr/images"

# -----------------------------
# Model files
# -----------------------------
MODEL_DIR = "/home/usr/models"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

# -----------------------------
# Camera / inference
# -----------------------------
CAM_SIZE = (1280, 720)
DNN_SIZE = (300, 300)
HEADER_H = 90

# -----------------------------
# Behavior toggles
# -----------------------------
DEBUG = True  # set False for normal operation

# Event policy: "person detected" => event log line
EVENT_CLASS = "person"
EVENT_MIN_CONF = 0.50  # independent of primary threshold (can be same)

# -----------------------------
# Classes
# -----------------------------
CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

PRIMARY_CLASSES = {"person", "dog", "cat", "car", "bus", "motorbike", "bicycle"}
SECONDARY_CLASSES = {"bird", "chair", "sofa", "bottle", "tvmonitor"}

PRIMARY_MIN_CONF = 0.50
SECONDARY_MIN_CONF = 0.60

# -----------------------------
# Overlay style (BGR)
# -----------------------------
COLOR_PRIMARY = (0, 255, 0)       # green
COLOR_SECONDARY = (0, 255, 255)   # yellow
COLOR_TEXT = (255, 255, 255)      # white
COLOR_DIM = (200, 200, 200)       # light gray


def pick_out_dir():
    out_dir = os.path.join(MOUNT, OUT_SUBDIR) if os.path.ismount(MOUNT) else FALLBACK_DIR
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def fail_if_missing(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing file: {path}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_primary(lbl: str) -> bool:
    return lbl in PRIMARY_CLASSES


def is_secondary(lbl: str) -> bool:
    return lbl in SECONDARY_CLASSES


def min_conf_for(lbl: str) -> float:
    if is_primary(lbl):
        return PRIMARY_MIN_CONF
    if is_secondary(lbl):
        return SECONDARY_MIN_CONF
    return 1.0  # filtered out


def draw_header(img_bgr, ts: str, dets):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

    cv2.putText(out, "AI DETECTION (MobileNet-SSD)", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(out, ts, (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_DIM, 2, cv2.LINE_AA)

    top = sorted(dets, key=lambda d: d["confidence"], reverse=True)[:3]
    summary = " | ".join([f"{d['label']} {int(round(d['confidence']*100))}%" for d in top])
    if not summary:
        summary = "no detections"

    (tw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x = max(20, w - 20 - tw)
    cv2.putText(out, summary, (x, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)

    return out


def draw_label(out, x1, y1, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    y = max(HEADER_H + th + 10, y1)
    cv2.rectangle(out, (x1, y - th - 10), (x1 + tw + 10, y), (0, 0, 0), -1)
    cv2.putText(out, text, (x1 + 5, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT, 2, cv2.LINE_AA)


def append_event(out_dir: str, line: str):
    path = os.path.join(out_dir, "events.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def main():
    # Fail fast on infrastructure
    fail_if_missing(PROTOTXT)
    fail_if_missing(CAFFEMODEL)

    out_dir = pick_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    overlay_img_path = os.path.join(out_dir, f"image_{ts}.jpg")
    meta_path = os.path.join(out_dir, f"meta_{ts}.json")

    raw_path = os.path.join(out_dir, f"raw_{ts}.jpg")
    dnn_in_path = os.path.join(out_dir, f"dnn_{ts}.jpg")

    # Hashing/version stamping
    prototxt_sha256 = sha256_file(PROTOTXT)
    caffemodel_sha256 = sha256_file(CAFFEMODEL)
    versions = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "platform": platform.platform(),
    }

    # Load network
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Capture
    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(main={"size": CAM_SIZE})
    picam2.configure(cfg)
    picam2.start()
    frame = picam2.capture_array()
    picam2.stop()

    if frame.ndim == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        raise SystemExit("ERROR: unexpected frame format from Picamera2")

    if DEBUG:
        cv2.imwrite(raw_path, frame_bgr)

    H, W = frame_bgr.shape[:2]

    # Prepare DNN input (and save it in debug mode)
    dnn_img = cv2.resize(frame_bgr, DNN_SIZE, interpolation=cv2.INTER_AREA)
    if DEBUG:
        cv2.imwrite(dnn_in_path, dnn_img)

    blob = cv2.dnn.blobFromImage(
        dnn_img,
        scalefactor=0.007843,   # 1/127.5
        size=DNN_SIZE,
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False,
    )

    net.setInput(blob)
    detections = net.forward()  # [1,1,N,7]

    dets = []
    out = frame_bgr.copy()

    person_event = None  # set if we see person over EVENT_MIN_CONF

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        cls_id = int(detections[0, 0, i, 1])

        if cls_id < 0 or cls_id >= len(CLASSES):
            continue

        label = CLASSES[cls_id]

        # Filter to confirmed sets only
        if not (is_primary(label) or is_secondary(label)):
            continue

        # Apply per-tier threshold
        if conf < min_conf_for(label):
            continue

        # Box (normalized)
        x1 = int(detections[0, 0, i, 3] * W)
        y1 = int(detections[0, 0, i, 4] * H)
        x2 = int(detections[0, 0, i, 5] * W)
        y2 = int(detections[0, 0, i, 6] * H)

        # Clamp + keep below header
        x1 = max(0, min(W - 1, x1))
        y1 = max(HEADER_H, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(HEADER_H, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        tier = "primary" if is_primary(label) else "secondary"
        color = COLOR_PRIMARY if tier == "primary" else COLOR_SECONDARY

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {int(round(conf * 100))}%"
        draw_label(out, x1, y1, tag)

        det = {
            "label": label,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "tier": tier,
        }
        dets.append(det)

        # Event trigger: first/highest-confidence person above EVENT_MIN_CONF
        if label == EVENT_CLASS and conf >= EVENT_MIN_CONF:
            if person_event is None or conf > person_event["confidence"]:
                person_event = det

    # Header
    out = draw_header(out, ts, dets)

    # Always write overlay image + JSON (even if no detections)
    if not cv2.imwrite(overlay_img_path, out):
        raise SystemExit("ERROR: failed to write overlay image")

    # One-line event log when person detected
    if person_event is not None:
        # log format: ISO8601 | event | conf | bbox | file
        bbox = person_event["bbox_xyxy"]
        line = (
            f"{datetime.now().isoformat(timespec='seconds')} | "
            f"PERSON | {person_event['confidence']:.3f} | "
            f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} | "
            f"{os.path.basename(overlay_img_path)}"
        )
        append_event(out_dir, line)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "output_dir": out_dir,
                "files": {
                    "overlay_image": os.path.basename(overlay_img_path),
                    "raw_image": os.path.basename(raw_path) if DEBUG else None,
                    "dnn_input_image": os.path.basename(dnn_in_path) if DEBUG else None,
                    "event_log": "events.log" if person_event is not None else None,
                },
                "camera": {
                    "size": list(CAM_SIZE),
                },
                "dnn": {
                    "input_size": list(DNN_SIZE),
                    "model": {
                        "prototxt": PROTOTXT,
                        "caffemodel": CAFFEMODEL,
                        "sha256": {
                            "prototxt": prototxt_sha256,
                            "caffemodel": caffemodel_sha256,
                        },
                    },
                },
                "classes": {
                    "primary": sorted(PRIMARY_CLASSES),
                    "secondary": sorted(SECONDARY_CLASSES),
                    "thresholds": {
                        "primary_min_conf": PRIMARY_MIN_CONF,
                        "secondary_min_conf": SECONDARY_MIN_CONF,
                        "event_class": EVENT_CLASS,
                        "event_min_conf": EVENT_MIN_CONF,
                    },
                },
                "versions": versions,
                "detections": dets,
                "event": {
                    "triggered": person_event is not None,
                    "best_person": person_event,
                },
                "debug": DEBUG,
            },
            f,
            indent=2,
        )

    print(overlay_img_path)


if __name__ == "__main__":
    main()