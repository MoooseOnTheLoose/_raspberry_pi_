# Write your code here :-)
#!/usr/bin/env python3
# rpiVIDSecAI.py
# Continuous recording in fixed 10-minute clips + AI detection + HUMAN_ENTER/HUMAN_EXIT logging
# Includes:
# - continuous back-to-back 10-minute MP4 clips
# - AI inference on lores stream (MobileNet-SSD) throttled
# - burned-in overlay with latest detections
# - event log lines include:
#     - clip filename
#     - clip-relative timestamp (seconds from clip start)
#
# Requires:
#   sudo apt install ffmpeg python3-picamera2 python3-opencv
# Model files:
#   /home/fry/models/MobileNetSSD_deploy.prototxt
#   /home/fry/models/MobileNetSSD_deploy.caffemodel

import os
import time
import json
import shutil
import hashlib
import platform
from datetime import datetime
from threading import Lock

import cv2
import numpy as np

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# -----------------------------
# Storage / logging
# -----------------------------
MOUNT = "/media/user/disk"
VIDEO_SUBDIR = "videos"
FALLBACK_DIR = "/home/user/videos"
EVENT_LOG_NAME = "events.log"
SESSION_META_NAME = "session_meta.json"

MIN_FREE_GB = 100

# -----------------------------
# Clips
# -----------------------------
CLIP_SECONDS = 600          # 10 minutes
FPS = 15
BITRATE = 2_000_000         # 2 Mbps

# -----------------------------
# Camera configuration
# -----------------------------
MAIN_SIZE = (1280, 720)     # recorded video
LORES_SIZE = (320, 240)     # inference stream
HEADER_H = 70

# -----------------------------
# AI model (Caffe MobileNet-SSD)
# -----------------------------
MODEL_DIR = "/home/usr/models"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

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

# Human logging policy
EVENT_CLASS = "person"
EVENT_MIN_CONF = 0.50

# State machine to avoid spam:
# - ENTER logged immediately on first confident person detection after absence.
# - EXIT logged when no person has been seen for a hold time.
PERSON_EXIT_HOLD = 8.0
PERSON_REENTER_COOLDOWN = 2.0

# Inference throttle
INFER_EVERY_N_FRAMES = 8    # at 15 fps ~ 1.9 inferences/sec

# DNN input
DNN_SIZE = (300, 300)
DNN_SCALE = 0.007843
DNN_MEAN = (127.5, 127.5, 127.5)

# Overlay colors (BGR)
COLOR_PRIMARY = (0, 255, 0)
COLOR_SECONDARY = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_DIM = (200, 200, 200)


def pick_out_dir():
    out_dir = os.path.join(MOUNT, VIDEO_SUBDIR) if os.path.ismount(MOUNT) else FALLBACK_DIR
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
    return 1.0


def append_event(out_dir: str, line: str):
    path = os.path.join(out_dir, EVENT_LOG_NAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def draw_label(img, x1, y1, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y = max(HEADER_H + th + 8, y1)
    cv2.rectangle(img, (x1, y - th - 8), (x1 + tw + 10, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2, cv2.LINE_AA)
                
def make_ffmpeg_output(mp4_path: str):
    try:
        return FfmpegOutput(mp4_path, audio=False, options=["-movflags", "+faststart"])
    except TypeError:
        try:
            return FfmpegOutput(mp4_path, audio=False)
        except:
            return FfmpegOutput(mp4_path)


def main():
    fail_if_missing(PROTOTXT)
    fail_if_missing(CAFFEMODEL)

    out_dir = pick_out_dir()
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # session metadata (hashing/version stamping)
    session_meta = {
        "session_ts": session_ts,
        "out_dir": out_dir,
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "model": {
            "prototxt": PROTOTXT,
            "caffemodel": CAFFEMODEL,
            "sha256": {
                "prototxt": sha256_file(PROTOTXT),
                "caffemodel": sha256_file(CAFFEMODEL),
            },
        },
        "settings": {
            "clip_seconds": CLIP_SECONDS,
            "fps": FPS,
            "bitrate": BITRATE,
            "main_size": list(MAIN_SIZE),
            "lores_size": list(LORES_SIZE),
            "infer_every_n_frames": INFER_EVERY_N_FRAMES,
            "thresholds": {
                "primary_min_conf": PRIMARY_MIN_CONF,
                "secondary_min_conf": SECONDARY_MIN_CONF,
                "event_class": EVENT_CLASS,
                "event_min_conf": EVENT_MIN_CONF,
            },
            "human_logging": {
                "exit_hold_s": PERSON_EXIT_HOLD,
                "reenter_cooldown_s": PERSON_REENTER_COOLDOWN,
            },
        },
    }
    with open(os.path.join(out_dir, SESSION_META_NAME), "w") as f:
        json.dump(session_meta, f, indent=2)

    # Load DNN once
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Shared overlay state
    det_lock = Lock()
    last_dets = []
    last_summary = "no detections"

    # Human presence state
    person_present = False
    last_person_seen_t = 0.0
    last_transition_t = 0.0

    # Current clip tracking (for event logs)
    current_clip_name = "unknown.mp4"
    clip_start_t = 0.0  # monotonic start time of current clip

    def clip_offset_s(now_t: float) -> float:
        # seconds from clip start (monotonic, not wall clock)
        if clip_start_t <= 0:
            return 0.0
        return max(0.0, now_t - clip_start_t)

    # Camera config (main for recording, lores for inference)
    picam2 = Picamera2()
    video_cfg = picam2.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam2.configure(video_cfg)

    def pre_callback(request):
        # overlay latest detections onto each frame
        with MappedArray(request, "main") as m:
            rgb = m.array
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]

            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

            now_s = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.putText(img, "AI VIDEO (MobileNet-SSD)", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, now_s, (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_DIM, 2, cv2.LINE_AA)

            with det_lock:
                dets = list(last_dets)
                summary = str(last_summary)

            (tw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(img, summary, (max(16, w - 16 - tw), 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT, 2, cv2.LINE_AA)

            for d in dets[:20]:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                lbl = d["label"]
                conf = d["confidence"]
                color = COLOR_PRIMARY if is_primary(lbl) else COLOR_SECONDARY
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                draw_label(img, x1, y1, f"{lbl} {int(round(conf * 100))}%")

            m.array[:, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    picam2.pre_callback = pre_callback

    encoder = H264Encoder(bitrate=BITRATE, framerate=FPS)

    def run_inference():
        nonlocal last_dets, last_summary
        nonlocal person_present, last_person_seen_t, last_transition_t
        nonlocal current_clip_name, clip_start_t

        lo = picam2.capture_array("lores")  # RGB
        lo_bgr = cv2.cvtColor(lo, cv2.COLOR_RGB2BGR)

        dnn_img = cv2.resize(lo_bgr, DNN_SIZE, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            dnn_img,
            scalefactor=DNN_SCALE,
            size=DNN_SIZE,
            mean=DNN_MEAN,
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        detections = net.forward()

        lo_h, lo_w = lo_bgr.shape[:2]
        main_w, main_h = MAIN_SIZE[0], MAIN_SIZE[1]
        sx = main_w / lo_w
        sy = main_h / lo_h

        dets = []
        best_person = None

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls_id = int(detections[0, 0, i, 1])
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue
            label = CLASSES[cls_id]

            if not (is_primary(label) or is_secondary(label)):
                continue
            if conf < min_conf_for(label):
                continue

            x1 = int(detections[0, 0, i, 3] * lo_w * sx)
            y1 = int(detections[0, 0, i, 4] * lo_h * sy)
            x2 = int(detections[0, 0, i, 5] * lo_w * sx)
            y2 = int(detections[0, 0, i, 6] * lo_h * sy)

            x1 = max(0, min(main_w - 1, x1))
            y1 = max(HEADER_H, min(main_h - 1, y1))
            x2 = max(0, min(main_w - 1, x2))
            y2 = max(HEADER_H, min(main_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            d = {"label": label, "confidence": conf, "bbox_xyxy": [x1, y1, x2, y2]}
            dets.append(d)

            if label == EVENT_CLASS and conf >= EVENT_MIN_CONF:
                if best_person is None or conf > best_person["confidence"]:
                    best_person = d

        dets.sort(key=lambda d: d["confidence"], reverse=True)
        top = dets[:3]
        summary = " | ".join([f"{d['label']} {int(round(d['confidence'] * 100))}%" for d in top]) if top else "no detections"

        with det_lock:
            last_dets = dets
            last_summary = summary

        now_t = time.time()

        if best_person is not None:
            last_person_seen_t = now_t

        # ENTER
        if (best_person is not None) and (not person_present) and (now_t - last_transition_t >= PERSON_REENTER_COOLDOWN):
            person_present = True
            last_transition_t = now_t
            bbox = best_person["bbox_xyxy"]
            off = clip_offset_s(now_t)
            line = (
                f"{datetime.now().isoformat(timespec='seconds')} | "
                f"HUMAN_ENTER | clip={current_clip_name} | "
                f"t={off:.2f}s | {best_person['confidence']:.3f} | "
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            )
            append_event(out_dir, line)

        # EXIT
        if person_present and (now_t - last_person_seen_t >= PERSON_EXIT_HOLD):
            person_present = False
            last_transition_t = now_t
            off = clip_offset_s(now_t)
            line = (
                f"{datetime.now().isoformat(timespec='seconds')} | "
                f"HUMAN_EXIT | clip={current_clip_name} | t={off:.2f}s"
            )
            append_event(out_dir, line)

    # Start camera once; loop clips forever
    picam2.start()

    seq = 0
    try:
        while True:
            if free_gb(out_dir) < MIN_FREE_GB:
                append_event(out_dir, f"{datetime.now().isoformat(timespec='seconds')} | STOP | low_disk")
                break

            clip_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_clip_name = f"secAI_{session_ts}_{seq:04d}_{clip_ts}.mp4"
            mp4_path = os.path.join(out_dir, current_clip_name)

            output = FfmpegOutput(mp4_path)

            # Start recording + set clip start reference time for offsets
            clip_start_t = time.time()
            picam2.start_recording(encoder, output)

            start_t = clip_start_t
            frame_count = 0

            while (time.time() - start_t) < CLIP_SECONDS:
                time.sleep(1.0 / FPS)
                frame_count += 1

                if frame_count % INFER_EVERY_N_FRAMES == 0:
                    run_inference()

                if free_gb(out_dir) < MIN_FREE_GB:
                    break

            picam2.stop_recording()
            seq += 1

            if free_gb(out_dir) < MIN_FREE_GB:
                append_event(out_dir, f"{datetime.now().isoformat(timespec='seconds')} | STOP | low_disk")
                break

    except KeyboardInterrupt:
        append_event(out_dir, f"{datetime.now().isoformat(timespec='seconds')} | STOP | ctrl_c")
    finally:
        try:
            if picam2.recording:
                picam2.stop_recording()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
