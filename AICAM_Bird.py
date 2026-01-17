#!/usr/bin/env python3
"""
Bird-feeder recorder (Raspberry Pi OS, Picamera2)

- Runs MobileNet-SSD (Caffe) inference on the lores stream.
- Starts recording only when "bird" is detected with sufficient confidence.
- Stops after a hold time of consecutive misses (post-roll).
- Prepends a short pre-roll using Picamera2 CircularOutput.
- Writes:
  - MP4 clip(s)
  - per-clip JSON metadata
  - events.log

Deps:
  sudo apt install -y ffmpeg python3-picamera2 python3-opencv
Model files (MobileNet-SSD Caffe):
  deploy.prototxt
  mobilenet_iter_73000.caffemodel
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FileOutput


# -----------------------------
# Logging
# -----------------------------
import logging
from logging.handlers import RotatingFileHandler

def _setup_logger(base: str) -> logging.Logger:
    log_path = os.path.join(base, EVENT_LOG_NAME)
    logger = logging.getLogger("birdcam")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# -----------------------------
# Storage
# -----------------------------
MOUNT = "/media/fry/deskView"
BASE_SUBDIR = "birdcam"
FALLBACK_DIR = "/home/fry/birdcam"
TMP_SUBDIR = "tmp"
EVENT_LOG_NAME = "events.log"

# -----------------------------
# Recording
# -----------------------------
FPS = 15
MAIN_SIZE = (1920, 1080)
LORES_SIZE = (640, 360)

BITRATE = 4_000_000  # H.264 encoder bitrate (bits/sec)

# Bird detection / gating
BIRD_MIN_CONF = 0.60
DETECT_HZ = 5.0          # inference ticks per second (lores grabs)
START_FRAMES = 3           # must see bird this many inference ticks to start recording
STOP_FRAMES = 10           # must miss bird this many inference ticks to stop recording
COOLDOWN_SEC = 5.0         # minimum seconds between clip starts
PRE_ROLL_SEC = 3.0         # seconds prepended before trigger (circular buffer)
MIN_CLIP_SEC = 2.0         # discard clips shorter than this
MAX_CLIP_SEC = 60.0        # split long recordings into multiple files

DETECT_INTERVAL_S = 1.0 / float(DETECT_HZ)

# -----------------------------
# Overlay
# -----------------------------
_toggle = False
HEADER_H = 76
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (160, 160, 160)
COLOR_OK = (80, 220, 80)
COLOR_WARN = (40, 180, 255)

# -----------------------------
# AI model (Caffe MobileNet-SSD)
# -----------------------------
MODEL_DIR = "/home/fry/models"  # <-- change if needed
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _fail_if_missing(path: str) -> None:
    if not os.path.exists(path):
        print(f"[FATAL] Missing required file: {path}", file=sys.stderr)
        sys.exit(2)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _pick_base_dir() -> str:
    """
    Prefer removable mount if present; else fallback to local.
    Creates:
      <base>/clips/YYYY/MM/DD
      <base>/meta/YYYY/MM/DD
      <base>/tmp
    """
    base = os.path.join(MOUNT, BASE_SUBDIR) if os.path.isdir(MOUNT) else FALLBACK_DIR

    day_dir = datetime.now().strftime("%Y/%m/%d")
    _ensure_dir(os.path.join(base, "clips", day_dir))
    _ensure_dir(os.path.join(base, "meta", day_dir))
    _ensure_dir(os.path.join(base, TMP_SUBDIR))
    return base

def _ffmpeg_remux_h264_to_mp4(src_h264: str, dst_mp4: str, fps: int) -> None:
    # -fflags +genpts is important for raw H.264
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-r", str(fps),
        "-fflags", "+genpts",
        "-i", src_h264,
        "-c", "copy",
        "-movflags", "+faststart",
        dst_mp4,
    ]
    subprocess.run(cmd, check=True)

def _ffmpeg_concat_mp4(mp4_paths: List[str], dst_mp4: str, work_dir: str) -> None:
    """
    Concatenate MP4s via concat demuxer. MP4s must have compatible streams.
    """
    list_path = os.path.join(work_dir, f"concat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in mp4_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        dst_mp4,
    ]
    subprocess.run(cmd, check=True)
    try:
        os.remove(list_path)
    except OSError:
        pass

def _detect_bird_on_lores(net, lo_bgr: np.ndarray) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    """
    Run MobileNet-SSD on a lores BGR frame.
    Returns (seen, best_conf, best_bbox_lores).
    """
    lo_h, lo_w = lo_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(lo_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_bbox = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        cls_id = int(detections[0, 0, i, 1])
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue
        if CLASSES[cls_id] != "bird":
            continue
        if conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([lo_w, lo_h, lo_w, lo_h])
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, min(lo_w - 1, x1))
            y1 = max(0, min(lo_h - 1, y1))
            x2 = max(0, min(lo_w - 1, x2))
            y2 = max(0, min(lo_h - 1, y2))
            best_bbox = (x1, y1, x2, y2)
            best_conf = conf

    return (best_conf >= BIRD_MIN_CONF), best_conf, best_bbox

def _setup_camera(latest_det: Dict[str, Any]) -> Tuple[Picamera2, H264Encoder, CircularOutput]:
    """
    Configure Picamera2 with:
      - main RGB for overlay
      - lores RGB for inference
      - H264 encoder writing into CircularOutput (pre-roll buffer)
    """

    picam2 = Picamera2()
    video_cfg = picam2.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam2.configure(video_cfg)

    encoder = H264Encoder(bitrate=BITRATE)

    # Picamera2 CircularOutput buffersize is in frames
    buffersize = int(max(1, round(PRE_ROLL_SEC * FPS))) + int(FPS * 2)
    circular = CircularOutput(buffersize=buffersize)

    def pre_callback(request):
        with MappedArray(request, "main") as m:
            img = m.array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

            now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img, "BIRDCAM (bird-feeder)", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, now_s, (16, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_DIM, 2, cv2.LINE_AA)

            # Detection overlay
            seen = bool(latest_det.get("seen", False))
            conf = float(latest_det.get("conf", 0.0))
            bbox = latest_det.get("bbox", None)

            status = f"BIRD {conf:.2f}" if seen else f"NO BIRD {conf:.2f}"
            color = COLOR_OK if seen else COLOR_WARN
            cv2.putText(img, status, (w - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            m.array[:, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    picam2.pre_callback = pre_callback
    return picam2, encoder, circular

def _start_circular(picam2: Picamera2, encoder: H264Encoder, circular: CircularOutput) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, circular)

def _start_file(picam2: Picamera2, encoder: H264Encoder, out_h264: str) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, FileOutput(out_h264))

def _finalize_clip(
    base: str,
    logger: logging.Logger,
    clip_name: str,
    tmp_pre_h264: Optional[str],
    tmp_main_h264: str,
    start_t: float,
    end_t: float,
    conf_values: List[float],
    det_count: int,
    note: str,
) -> None:
    day_dir = datetime.now().strftime("%Y/%m/%d")
    clips_dir = os.path.join(base, "clips", day_dir)
    meta_dir = os.path.join(base, "meta", day_dir)
    tmp_dir = os.path.join(base, TMP_SUBDIR)

    _ensure_dir(clips_dir)
    _ensure_dir(meta_dir)
    _ensure_dir(tmp_dir)

    duration_s = max(0.0, end_t - start_t)

    final_mp4 = os.path.join(clips_dir, f"{clip_name}.mp4")
    tmp_main_mp4 = os.path.join(tmp_dir, f"{clip_name}_main.mp4")
    tmp_pre_mp4 = os.path.join(tmp_dir, f"{clip_name}_pre.mp4")

    # Remux/concat
    _ffmpeg_remux_h264_to_mp4(tmp_main_h264, tmp_main_mp4, fps=FPS)
    if tmp_pre_h264 and os.path.exists(tmp_pre_h264):
        _ffmpeg_remux_h264_to_mp4(tmp_pre_h264, tmp_pre_mp4, fps=FPS)
        _ffmpeg_concat_mp4([tmp_pre_mp4, tmp_main_mp4], final_mp4, work_dir=tmp_dir)
    else:
        shutil.move(tmp_main_mp4, final_mp4)

    # Metadata
    conf_min = float(min(conf_values)) if conf_values else 0.0
    conf_max = float(max(conf_values)) if conf_values else 0.0
    conf_mean = float(sum(conf_values) / len(conf_values)) if conf_values else 0.0

    meta = {
        "clip_name": clip_name,
        "mp4_path": final_mp4,
        "start_local": datetime.fromtimestamp(start_t).astimezone().isoformat(timespec="seconds"),
        "start_utc": datetime.fromtimestamp(start_t, tz=timezone.utc).isoformat(timespec="seconds"),
        "end_local": datetime.fromtimestamp(end_t).astimezone().isoformat(timespec="seconds"),
        "end_utc": datetime.fromtimestamp(end_t, tz=timezone.utc).isoformat(timespec="seconds"),
        "duration_s": float(duration_s),
        "fps": int(FPS),
        "main_size": list(MAIN_SIZE),
        "lores_size": list(LORES_SIZE),
        "bird_min_conf": float(BIRD_MIN_CONF),
        "detections": int(det_count),
        "conf_min": conf_min,
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "note": note,
    }

    meta_path = os.path.join(meta_dir, f"{clip_name}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Cleanup h264 + temps
    for p in [tmp_pre_h264, tmp_main_h264, tmp_pre_mp4, tmp_main_mp4]:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    if duration_s < MIN_CLIP_SEC:
        # discard final mp4 too
        try:
            os.remove(final_mp4)
        except OSError:
            pass
        logger.info(f"DISCARD | clip={clip_name} | dur={duration_s:.2f}s {note}")
    else:
        logger.info(f"CLIP | clip={clip_name} | dur={duration_s:.2f}s {note}")

def main() -> None:
    _fail_if_missing(PROTOTXT)
    _fail_if_missing(CAFFEMODEL)

    base = _pick_base_dir()
    logger = _setup_logger(base)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"START | session={session_ts}")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Shared detection snapshot for overlay
    latest_det: Dict[str, Any] = {"seen": False, "conf": 0.0, "bbox": None, "ts": 0.0}

    picam2, encoder, circular = _setup_camera(latest_det)
    picam2.start()
    _start_circular(picam2, encoder, circular)

    recording = False
    last_clip_end_mono = 0.0  # monotonic

    seen_streak = 0
    absent_streak = 0

    det_count = 0
    conf_values: List[float] = []

    clip_name = ""
    clip_start_mono = 0.0     # monotonic
    clip_start_wall = 0.0     # wall-clock for metadata
    clip_tmp_pre_h264: Optional[str] = None
    clip_tmp_main_h264 = ""

    last_detect_mono = 0.0    # monotonic

    def start_clip(trigger_conf: float) -> None:
        nonlocal recording, last_clip_end_mono
        nonlocal det_count, conf_values
        nonlocal clip_name, clip_start_mono, clip_start_wall, clip_tmp_pre_h264, clip_tmp_main_h264

        now_mono = time.monotonic()
        if now_mono - last_clip_end_mono < COOLDOWN_SEC:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        clip_name = f"bird_{stamp}"
        clip_start_mono = now_mono
        clip_start_wall = time.time()

        tmp_dir = os.path.join(base, TMP_SUBDIR)
        _ensure_dir(tmp_dir)

        clip_tmp_pre_h264 = os.path.join(tmp_dir, f"{clip_name}_pre.h264")
        clip_tmp_main_h264 = os.path.join(tmp_dir, f"{clip_name}_main.h264")

        # Save pre-roll from circular buffer (if any)
        try:
            circular.stop()
        except Exception:
            pass
        try:
            circular.copy_to_file(clip_tmp_pre_h264)
        except Exception:
            clip_tmp_pre_h264 = None

        # Switch to direct-to-file recording for the main portion
        _start_file(picam2, encoder, clip_tmp_main_h264)

        recording = True
        det_count = 0
        conf_values = []
        logger.info(f"TRIGGER | clip={clip_name} | conf={trigger_conf:.2f}")

    def stop_clip(note: str) -> None:
        nonlocal recording, last_clip_end_mono
        nonlocal clip_name, clip_start_mono, clip_start_wall, clip_tmp_pre_h264, clip_tmp_main_h264
        nonlocal det_count, conf_values

        if not recording:
            return

        end_mono = time.monotonic()
        end_wall = time.time()

        try:
            picam2.stop_recording()
        except Exception:
            pass

        # Immediately return to circular buffer mode
        _start_circular(picam2, encoder, circular)

        recording = False
        last_clip_end_mono = end_mono

        _finalize_clip(
            base=base,
            logger=logger,
            clip_name=clip_name,
            tmp_pre_h264=clip_tmp_pre_h264,
            tmp_main_h264=clip_tmp_main_h264,
            start_t=clip_start_wall,
            end_t=end_wall,
            conf_values=conf_values,
            det_count=det_count,
            note=note,
        )

        clip_name = ""
        clip_start_mono = 0.0
        clip_start_wall = 0.0
        clip_tmp_pre_h264 = None
        clip_tmp_main_h264 = ""
        det_count = 0
        conf_values = []

    try:
        while True:
            now_mono = time.monotonic()

            # Detection cadence (monotonic)
            if now_mono - last_detect_mono < DETECT_INTERVAL_S:
                time.sleep(0.01)
                continue
            last_detect_mono = now_mono

            lo_rgb = picam2.capture_array("lores")
            lo_bgr = cv2.cvtColor(lo_rgb, cv2.COLOR_RGB2BGR)

            seen, conf, bbox_lo = _detect_bird_on_lores(net, lo_bgr)

            # Map bbox to main resolution for overlay display
            bbox_main = None
            if bbox_lo is not None:
                lo_h, lo_w = lo_bgr.shape[:2]
                main_w, main_h = MAIN_SIZE
                sx = main_w / float(lo_w)
                sy = main_h / float(lo_h)
                x1, y1, x2, y2 = bbox_lo
                bbox_main = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            latest_det.update({"seen": seen, "conf": float(conf), "bbox": bbox_main, "ts": time.time()})

            # Update streaks
            if seen:
                seen_streak += 1
                absent_streak = 0
            else:
                absent_streak += 1
                seen_streak = 0

            # Start when stable
            if (not recording) and (seen_streak >= START_FRAMES):
                start_clip(trigger_conf=float(conf))
                continue

            # While recording: track stats and stop conditions
            if recording:
                if seen:
                    det_count += 1
                    conf_values.append(float(conf))

                # Stop when stable absent
                if absent_streak >= STOP_FRAMES:
                    stop_clip(note="bird_absent")
                    continue

                # Split long clips (monotonic)
                if now_mono - clip_start_mono >= MAX_CLIP_SEC:
                    stop_clip(note="split_max_len")
                    continue

    except KeyboardInterrupt:
        logger.info(f"STOP | session={session_ts} | reason=keyboardinterrupt")
        try:
            stop_clip(note="shutdown")
        except Exception:
            pass
    except Exception as e:
        logger.exception("ERROR")
        try:
            stop_clip(note="crash")
        except Exception:
            pass
        raise
    finally:
        try:
            picam2.stop_recording()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()