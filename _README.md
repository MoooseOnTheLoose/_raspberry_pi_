# AICAM (Raspberry Pi) — Event-Driven Vision System

A **stateful, auditable, event-driven vision system** with **deterministic outputs** and **controlled resource ownership**.

This project is intentionally designed to:
- survive long runtimes
- produce explainable artifacts
- evolve without fragile coupling

---

## Core Design Philosophy

Signal>Frames
Events>Video
System>Scripts

The system emphasizes **reliable, explainable events** over raw media capture. Video and images are treated as **artifacts attached to events**, not the primary output.

---

## Architecture at a Glance

Two complementary intelligence layers exist in the project:

- **Tier 1:** IMX500 (on-sensor AI) — efficient, best for stills, but fragile metadata coupling.
- **Tier 2:** Picamera2 + OpenCV DNN — portable, debuggable, production-consistent (primary production path).

---

## Intelligence Layers

### Tier 1 — IMX500 (On-Sensor AI)

IMX500 (On-Sensor AI)
│
│  stdout / metadata
▼
Python (parse)
│
▼
OpenCV (draw)
│
▼
JPEG

**Strengths**
- Efficient, low power
- Great for stills
- Hardware-accelerated inference

**Tradeoffs**
- Fragile stdout/metadata coupling
- Best effort overlays depend on consistent metadata formats
- Single-owner constraints apply

---

### Tier 2 — Picamera2 + OpenCV DNN (Production-Consistent)

Picamera2
│
├── Main Stream (RGB)
│       │
│       ▼
│   Burn-in Overlay (OpenCV)
│       │
│       ▼
│   JPEG / MP4
│
└── Lores Stream (RGB)
│
▼
OpenCV DNN (optional ML)
│
▼
Detection Results

**Strengths**
- Portable and debuggable
- Stable for long-running services
- Clean separation between capture / inference / overlay / events
- Deterministic, auditable outputs

---

## Critical Rule: Controlled Resource Ownership

ONE CAMERA OWNER
start → run → stop (finally)
NO parallel access

This rule prevents:
- `pipeline handler in use by another process`
- race conditions in libcamera / Picamera2
- unpredictable capture failures

---

## Applications

### AICAM_photo — Still Capture + Detection

Still Capture
│
▼
OpenCV DNN
│
▼
Overlay Render
│
├── JPEG (annotated image)
├── JSON (detections + metadata)
└── Event Log (if person detected)

**What it produces**
- A single annotated image (JPEG)
- A structured metadata record (JSON)
- An append-only event log entry (when policy triggers)

---

### AICAM_VIDSec — Continuous Video Security Recorder

Video Stream
│
▼
OpenCV DNN (throttled)
│
▼
State Machine
(person present?)
│
├── HUMAN_ENTER → log event
├── HUMAN_EXIT  → log event
│
▼
Burn-in Overlay
│
▼
Segmented MP4 Clips (10 min)

**What it produces**
- Continuous **10-minute MP4 clips**
- Burned-in overlay (boxes + labels + confidence)
- Event log entries with clip + clip-relative timestamps

---

## State & Policy

The system operates as a **state machine**, not a frame-by-frame logger.

STATE:
person_present = True / False

POLICY:
HUMAN_ENTER → log + annotate
HUMAN_EXIT  → log only

**Why it matters**
- prevents event spam
- makes logs meaningful for review
- enables “security-style” behavior (enter/exit transitions)

---

## Determinism & Traceability (Audit Layer)

Deterministic Outputs
├── Model SHA256 hashes
├── OpenCV / Python / OS versions
├── Clip-relative timestamps
└── Append-only event logs

This ensures:
- reproducibility across machines
- traceable changes when performance shifts
- credible, explainable records over time

---

## Output Artifacts

### Directory selection
Outputs are written to:
- Preferred: `/media/user/disk/...` (when mounted)
- Fallback: `/home/user/...`

### Typical artifacts

**Still mode**
- `image_<ts>.jpg` (annotated)
- `meta_<ts>.json` (detections + versions + hashes)
- `events.log` (append-only)

**Video mode**
- `secAI_<session>_<seq>_<ts>.mp4` (10-minute segments)
- `events.log` containing:
  - `HUMAN_ENTER` / `HUMAN_EXIT`
  - `clip=<filename>`
  - `t=<seconds>` (clip-relative timestamp)
  - confidence + bbox for enter events

---

## Production Compatibility Notes

### Picamera2 API drift (FFmpeg output)
Picamera2 versions vary. This project uses **capability detection**, not brittle version checks:

```python
def make_ffmpeg_output(mp4_path: str):
    try:
        return FfmpegOutput(mp4_path, audio=False, options=["-movflags", "+faststart"])
    except TypeError:
        try:
            return FfmpegOutput(mp4_path, audio=False)
        except TypeError:
            return FfmpegOutput(mp4_path)

