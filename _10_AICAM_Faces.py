#!/usr/bin/env python3
"""
FACECAM / HEADCAM (Raspberry Pi OS) — faces-only trigger recorder with pre-roll

What this is
- Raspberry Pi OS–friendly version aligned with HUMANCAM declarations/patterns.
- Uses Picamera2 + H264Encoder + CircularOutput (encoded ring buffer) for pre-roll.
- Uses OpenCV DNN Caffe face detector (no pip / no ultralytics).
- Writes MP4 clips + per-clip JSON metadata + a simple events.log.

Requirements (Raspberry Pi OS apt)
  sudo apt install -y ffmpeg python3-picamera2 python3-opencv

Model files (OpenCV DNN Caffe face detector)
  deploy.prototxt
  res10_300x300_ssd_iter_140000.caffemodel

Place model files in MODEL_DIR below.

Notes
- Detection runs on the lores stream for efficiency.
- Pre-roll uses CircularOutput.copy_to(...) before switching to file output.
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, CircularOutput
from picamera2 import MappedArray


# -----------------------------
# Storage (match HUMANCAM pattern)
# -----------------------------
MOUNT = "/media/usr/disk"
BASE_SUBDIR = "facecam"
TMP_SUBDIR = "tmp"
FALLBACK_DIR = "/home/usr/facecam"

EVENT_LOG_NAME = "events.log"
MIN_FREE_GB = 10.0

# -----------------------------
# Recording parameters (match HUMANCAM style)
# -----------------------------
FPS = 15
BITRATE = 6000000

MAIN_SIZE = (1920, 1080)
LORES_SIZE = (640, 360)

# Target detection / gating
FACE_MIN_CONF = 0.05
INFER_EVERY_N_FRAMES = 3
DETECT_INTERVAL_S = INFER_EVERY_N_FRAMES / float(FPS)
START_FRAMES = 3
STOP_FRAMES = 10
COOLDOWN_SEC = 5.0
PRE_ROLL_SEC = 3.0
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 60.0

# Overlay
HEADER_H = 76
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (160, 160, 160)
COLOR_OK = (80, 220, 80)

# -----------------------------
# AI model (Caffe face detector)
# -----------------------------
MODEL_DIR = "/home/usr/models"  # <-- change if needed
PROTOTXT = os.path.join(MODEL_DIR, "deployFace.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

TARGET_LABEL = "face"

# -----------------------------
# ROI / exclusion zones (same semantics as HUMANCAM)
# -----------------------------
ROI_ENABLE = True
ROI_INCLUDE_POLY_NORM = None  # e.g. [(0.10,0.20),(0.90,0.20),(0.90,0.95),(0.10,0.95)]
ROI_EXCLUDE_POLYS_NORM = []   # e.g. [[(0,0),(1,0),(1,0.12),(0,0.12)]]


# -----------------------------
# Helpers (copied/adapted from HUMANCAM)
# -----------------------------

def fail_if_missing(path: str) -> None:
    if not os.path.exists(path):
        print(f"[FATAL] Missing required file: {path}", file=sys.stderr)
        sys.exit(2)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_event(base: str, line: str) -> None:
    try:
        with open(os.path.join(base, EVENT_LOG_NAME), "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass


def pick_out_dir() -> str:
    """Prefer mounted storage if available and healthy; else fallback."""
    cand = os.path.join(MOUNT, BASE_SUBDIR)
    try:
        if os.path.ismount(MOUNT) and free_gb(MOUNT) >= MIN_FREE_GB:
            ensure_dir(cand)
            ensure_dir(os.path.join(cand, TMP_SUBDIR))
            return cand
    except Exception:
        pass

    ensure_dir(FALLBACK_DIR)
    ensure_dir(os.path.join(FALLBACK_DIR, TMP_SUBDIR))
    return FALLBACK_DIR


def ffmpeg_remux_h264_to_mp4(in_h264: str, out_mp4: str, fps: int) -> None:
    """Remux raw H.264 bitstream to MP4 without re-encoding."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-r",
        str(int(fps)),
        "-i",
        in_h264,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        out_mp4,
    ]
    subprocess.check_call(cmd)


def ffmpeg_concat_mp4(inputs: List[str], out_mp4: str) -> None:
    """Concatenate MP4 files losslessly via concat demuxer."""
    tmp_list = out_mp4 + ".concat.txt"
    with open(tmp_list, "w", encoding="utf-8") as f:
        for p in inputs:
            f.write(f"file '{p}'\n")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        tmp_list,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        out_mp4,
    ]
    try:
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(tmp_list)
        except OSError:
            pass


# -----------------------------
# ROI helpers (same as HUMANCAM)
# -----------------------------

def _norm_poly_to_px(poly_norm, w: int, h: int):
    if not poly_norm:
        return None
    pts = []
    for x, y in poly_norm:
        x = int(round(float(x) * (w - 1)))
        y = int(round(float(y) * (h - 1)))
        pts.append((x, y))
    return pts


def _point_in_poly(pt, poly_pts) -> bool:
    if not poly_pts or len(poly_pts) < 3:
        return False
    arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(arr, pt, False) >= 0


def _roi_accept_bbox_center(bbox_lores, lo_w: int, lo_h: int) -> bool:
    if not ROI_ENABLE:
        return True
    if bbox_lores is None:
        return False
    x1, y1, x2, y2 = bbox_lores
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))

    inc = _norm_poly_to_px(ROI_INCLUDE_POLY_NORM, lo_w, lo_h)
    if inc and not _point_in_poly((cx, cy), inc):
        return False

    for poly_norm in (ROI_EXCLUDE_POLYS_NORM or []):
        exc = _norm_poly_to_px(poly_norm, lo_w, lo_h)
        if exc and _point_in_poly((cx, cy), exc):
            return False

    return True


def _roi_polys_for_main():
    lo_w, lo_h = LORES_SIZE
    main_w, main_h = MAIN_SIZE
    sx = main_w / float(lo_w)
    sy = main_h / float(lo_h)

    def scale(poly_norm):
        pts = _norm_poly_to_px(poly_norm, lo_w, lo_h)
        if not pts:
            return None
        return [(int(round(x * sx)), int(round(y * sy))) for x, y in pts]

    inc = scale(ROI_INCLUDE_POLY_NORM) if ROI_INCLUDE_POLY_NORM else None
    excs = []
    for poly_norm in (ROI_EXCLUDE_POLYS_NORM or []):
        s = scale(poly_norm)
        if s:
            excs.append(s)
    return inc, excs


# -----------------------------
# Clip metadata
# -----------------------------

@dataclass
class ClipMeta:
    clip_name: str
    clip_path: str
    meta_path: str
    tmp_main_h264: str
    tmp_pre_h264: Optional[str] = None

    fps: int = FPS
    pre_roll_s: float = 0.0

    trigger_class: str = TARGET_LABEL

    start_local: str = ""
    start_utc: str = ""
    end_local: str = ""
    end_utc: str = ""

    duration_s: float = 0.0
    detections: int = 0
    conf_min: float = 0.0
    conf_mean: float = 0.0
    conf_max: float = 0.0


def write_clip_meta(meta: ClipMeta) -> None:
    ensure_dir(os.path.dirname(meta.meta_path))
    with open(meta.meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, ensure_ascii=False)


def finalize_clip(base: str, meta: ClipMeta, conf_values: List[float], duration_s: float, note: str = "") -> None:
    final_mp4 = meta.clip_path
    tmp_dir = os.path.join(base, TMP_SUBDIR)
    ensure_dir(tmp_dir)

    tmp_pre_mp4 = os.path.join(tmp_dir, f"{meta.clip_name}_pre.mp4")
    tmp_main_mp4 = os.path.join(tmp_dir, f"{meta.clip_name}_main.mp4")

    try:
        if conf_values:
            meta.conf_min = float(min(conf_values))
            meta.conf_mean = float(sum(conf_values) / len(conf_values))
            meta.conf_max = float(max(conf_values))
            meta.detections = int(len(conf_values))

        ffmpeg_remux_h264_to_mp4(meta.tmp_main_h264, tmp_main_mp4, meta.fps)

        if meta.tmp_pre_h264 and os.path.exists(meta.tmp_pre_h264):
            ffmpeg_remux_h264_to_mp4(meta.tmp_pre_h264, tmp_pre_mp4, meta.fps)
            ffmpeg_concat_mp4([tmp_pre_mp4, tmp_main_mp4], final_mp4)
        else:
            shutil.move(tmp_main_mp4, final_mp4)

        meta.end_local = local_now_iso()
        meta.end_utc = utc_now_iso()
        meta.duration_s = float(duration_s)

        if duration_s < MIN_CLIP_SEC:
            try:
                os.remove(final_mp4)
            except OSError:
                pass
            append_event(base, f"{local_now_iso()} | DISCARD | clip={meta.clip_name} | dur={duration_s:.2f}s {note}".rstrip())
        else:
            write_clip_meta(meta)
            append_event(base, f"{local_now_iso()} | FACE_EXIT | clip={meta.clip_name} | dur={duration_s:.2f}s {note}".rstrip())

    except Exception as e:
        qdir = os.path.join(base, "quarantine")
        ensure_dir(qdir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if os.path.exists(final_mp4):
                shutil.move(final_mp4, os.path.join(qdir, f"{stamp}_{os.path.basename(final_mp4)}"))
        except Exception:
            pass
        append_event(base, f"{local_now_iso()} | ERROR | finalize | clip={meta.clip_name} | {type(e).__name__}: {e}")

    finally:
        for p in [meta.tmp_main_h264, meta.tmp_pre_h264, tmp_pre_mp4, tmp_main_mp4]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# -----------------------------
# Detection (face detector)
# -----------------------------

def detect_face_on_lores(net, lo_bgr: np.ndarray) -> Tuple[bool, str, float, Optional[Tuple[int, int, int, int]]]:
    lo_h, lo_w = lo_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(lo_bgr, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_bbox = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < FACE_MIN_CONF:
            continue

        box = detections[0, 0, i, 3:7] * np.array([lo_w, lo_h, lo_w, lo_h])
        (x1, y1, x2, y2) = box.astype("int")
        x1 = max(0, min(lo_w - 1, x1))
        y1 = max(0, min(lo_h - 1, y1))
        x2 = max(0, min(lo_w - 1, x2))
        y2 = max(0, min(lo_h - 1, y2))
        cand = (x1, y1, x2, y2)

        if not _roi_accept_bbox_center(cand, lo_w, lo_h):
            continue

        if conf > best_conf:
            best_conf = conf
            best_bbox = cand

    seen = best_conf >= FACE_MIN_CONF and best_bbox is not None
    return seen, TARGET_LABEL, best_conf, best_bbox


# -----------------------------
# Camera setup (overlay matches HUMANCAM style)
# -----------------------------

def draw_label(img_bgr, x: int, y: int, text: str, color) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x2 = min(img_bgr.shape[1] - 1, x + tw + 8)
    y2 = min(img_bgr.shape[0] - 1, y + th + 10)
    cv2.rectangle(img_bgr, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (x + 4, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def setup_camera_with_overlay(latest_det: Dict[str, Any]) -> Tuple[Picamera2, H264Encoder, CircularOutput]:
    picam2 = Picamera2()
    video_cfg = picam2.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam2.configure(video_cfg)

    encoder = H264Encoder(bitrate=BITRATE)

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
            cv2.putText(img, "FACECAM", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, now_s, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_DIM, 2, cv2.LINE_AA)

            det_age = time.time() - float(latest_det.get("ts", 0))
            conf = float(latest_det.get("conf", 0.0))
            seen = bool(latest_det.get("seen", False)) and det_age < 2.0

            status = f"{TARGET_LABEL}: {'YES' if seen else 'no'}  conf={conf:.2f}"
            cv2.putText(img, status, (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (COLOR_OK if seen else COLOR_DIM), 2, cv2.LINE_AA)

            if ROI_ENABLE:
                inc_poly, exc_polys = _roi_polys_for_main()
                if inc_poly:
                    cv2.polylines(img, [np.array(inc_poly, dtype=np.int32)], True, (255, 255, 0), 2)
                    cv2.putText(img, "ROI", (inc_poly[0][0] + 6, inc_poly[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                for ep in exc_polys:
                    cv2.polylines(img, [np.array(ep, dtype=np.int32)], True, (0, 0, 255), 2)
                    cv2.putText(img, "EXCLUDE", (ep[0][0] + 6, ep[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            if seen and latest_det.get("bbox") is not None:
                x1, y1, x2, y2 = latest_det["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OK, 2)
                draw_label(img, x1, y1, f"{TARGET_LABEL} {conf:.2f}", COLOR_OK)

            m.array[:] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    picam2.pre_callback = pre_callback
    return picam2, encoder, circular


def start_circular_recording(picam2: Picamera2, encoder: H264Encoder, circular: CircularOutput) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, circular)


def start_file_recording(picam2: Picamera2, encoder: H264Encoder, out_h264: str) -> None:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    picam2.start_recording(encoder, FileOutput(out_h264))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    fail_if_missing(PROTOTXT)
    fail_if_missing(CAFFEMODEL)

    base = pick_out_dir()
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    append_event(base, f"{local_now_iso()} | START | session={session_ts}")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    latest_det: Dict[str, Any] = {"seen": False, "conf": 0.0, "bbox": None, "ts": 0.0}

    picam2, encoder, circular = setup_camera_with_overlay(latest_det)
    picam2.start()

    start_circular_recording(picam2, encoder, circular)

    recording = False
    cooldown_until = 0.0
    seen_streak = 0
    absent_streak = 0

    current_meta: Optional[ClipMeta] = None
    conf_values: List[float] = []
    clip_start_t = 0.0

    def new_paths(clip_name: str) -> Tuple[str, str, str]:
        day_path = datetime.now().strftime("%Y/%m/%d")
        clip_dir = os.path.join(base, "clips", day_path)
        meta_dir = os.path.join(base, "meta", day_path)
        tmp_dir = os.path.join(base, TMP_SUBDIR)
        ensure_dir(clip_dir)
        ensure_dir(meta_dir)
        ensure_dir(tmp_dir)
        final_mp4 = os.path.join(clip_dir, f"{clip_name}.mp4")
        meta_json = os.path.join(meta_dir, f"{clip_name}.json")
        tmp_main_h264 = os.path.join(tmp_dir, f"{clip_name}_main.h264")
        return final_mp4, meta_json, tmp_main_h264

    def start_clip(with_preroll: bool, note: str = "") -> None:
        nonlocal recording, current_meta, conf_values, clip_start_t

        clip_name = "face_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        final_mp4, meta_json, tmp_main_h264 = new_paths(clip_name)
        tmp_pre_h264 = os.path.join(base, TMP_SUBDIR, f"{clip_name}_pre.h264")

        current_meta = ClipMeta(
            clip_name=clip_name,
            clip_path=final_mp4,
            meta_path=meta_json,
            tmp_main_h264=tmp_main_h264,
            tmp_pre_h264=None,
            pre_roll_s=0.0,
            trigger_class=TARGET_LABEL,
            start_local=local_now_iso(),
            start_utc=utc_now_iso(),
        )
        conf_values = []

        if with_preroll and PRE_ROLL_SEC > 0:
            try:
                circular.copy_to(tmp_pre_h264, seconds=PRE_ROLL_SEC)
                current_meta.tmp_pre_h264 = tmp_pre_h264
                current_meta.pre_roll_s = float(PRE_ROLL_SEC)
            except Exception:
                current_meta.tmp_pre_h264 = None
                current_meta.pre_roll_s = 0.0

        start_file_recording(picam2, encoder, tmp_main_h264)
        recording = True
        clip_start_t = time.time()
        append_event(base, f"{local_now_iso()} | FACE_ENTER | clip={clip_name} {note}".rstrip())

    def stop_clip(note: str = "") -> None:
        nonlocal recording, current_meta, conf_values, clip_start_t
        if not recording or current_meta is None:
            return

        try:
            picam2.stop_recording()
        except Exception:
            pass

        duration_s = max(0.0, time.time() - clip_start_t)
        finalize_clip(base, current_meta, conf_values, duration_s, note=note)

        recording = False
        current_meta = None
        conf_values = []
        clip_start_t = 0.0

        start_circular_recording(picam2, encoder, circular)

    try:
        last_detect_t = 0.0

        while True:
            now = time.time()

            if now - last_detect_t < DETECT_INTERVAL_S:
                time.sleep(0.01)
                continue
            last_detect_t = now

            lo_rgb = picam2.capture_array("lores")
            lo_bgr = cv2.cvtColor(lo_rgb, cv2.COLOR_RGB2BGR)

            seen, label, conf, bbox_lo = detect_face_on_lores(net, lo_bgr)

            bbox_main = None
            if bbox_lo is not None:
                lo_h, lo_w = lo_bgr.shape[:2]
                main_w, main_h = MAIN_SIZE
                sx = main_w / float(lo_w)
                sy = main_h / float(lo_h)
                x1, y1, x2, y2 = bbox_lo
                bbox_main = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            latest_det.update({"seen": seen, "conf": float(conf), "bbox": bbox_main, "ts": now})

            if seen:
                seen_streak += 1
                absent_streak = 0
            else:
                absent_streak += 1
                seen_streak = 0

            if recording and current_meta is not None and seen:
                conf_values.append(float(conf))

            if recording and current_meta is not None:
                seg_len = now - clip_start_t
                if seg_len >= MAX_CLIP_SEC:
                    stop_clip(note="segment_max")
                    start_clip(with_preroll=False, note="segment_continue")
                    continue

            if (not recording) and now >= cooldown_until and seen_streak >= START_FRAMES:
                start_clip(with_preroll=True)
                continue

            if recording and absent_streak >= STOP_FRAMES:
                stop_clip(note="target_gone")
                cooldown_until = time.time() + COOLDOWN_SEC
                continue

    except KeyboardInterrupt:
        append_event(base, f"{local_now_iso()} | STOP | KeyboardInterrupt")

    finally:
        try:
            if recording and current_meta is not None:
                stop_clip(note="shutdown")
        except Exception:
            pass
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
