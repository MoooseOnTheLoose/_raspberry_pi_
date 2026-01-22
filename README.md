# AICAM (Raspberry Pi) 

UPDATED: 1/21/2026

PLANNING: 
1. Single authoritative storage path selection logic
2. Unified logging configuration (paths, rotation, verbosity)
3. Consistent directory layout for all artifacts
4. Systemd units with explicit mount dependencies and sandboxing rules

#
Event-Driven Camera + Offline Vision Pipelines + Hardening

![image](https://github.com/user-attachments/assets/af873b00-07f1-4b2f-b84c-f61220c61cfd)



AICAM is a set of Raspberry Pi camera pipelines designed around a small number of strict principles:

- **Offline-first** — no cloud dependency or required network access
- **Event-first** — state transitions are first-class artifacts, not log noise
- **Deterministic outputs** — version stamps, hashes, and reproducible behavior where applicable
- **One camera owner at a time** — no parallel Picamera2 / libcamera access
---

## Project scope

This repository contains:

- Baseline `rpicam-*` recorders (stills, test clips, continuous segmented recording)
- Picamera2-based AI pipelines (MobileNet-SSD legacy; YOLO ONNX is the forward path)
- Evidence-oriented event logging and artifact generation
- Guides covering:
  - camera tuning
  - encrypted storage and outbox workflows
  - receiver ingest and verification
  - training and tuning notes (especially for drone detection)

The project is **not** a single monolithic application. It is a collection of cooperating pipelines
that share architectural rules.

---

## Repository layout

### Baseline camera scripts (no AI)

These scripts rely on the native `rpicam-*` tools and are primarily used for
hardware validation, storage testing, and simple capture workflows.

- `1_rpiCAMStill.py`  
  Single still image capture via `rpicam-still`.

- `2_rpiVIDTest.py`  
  Short test video capture via `rpicam-vid`.

- `3_rpiVIDSec.py`  
  Continuous segmented recording (10-minute clips) with a disk free-space guard.
  This script establishes the canonical mount + fallback behavior used as a reference.

- `4_rpiCAMStill.sh`  
  Shell wrapper for still capture.

- `5_rpiVIDTest.sh`  
  Shell wrapper for test video capture.

- `6_rpiVIDSec.sh`  
  Shell wrapper for segmented security recording.

---

### AI pipelines (Picamera2 + OpenCV DNN)

These pipelines perform on-device inference using Picamera2 and OpenCV.

- `7_AICAM_Still.py`  
  Still capture + MobileNet-SSD (Caffe) inference, annotated output,
  and a single-line event record.

- `8_AICAM_VIDSec.py`  
  Continuous video recording with MobileNet-SSD inference and explicit
  `HUMAN_ENTER` / `HUMAN_EXIT` state transitions.

- `9_AICAM_Humans.py`  
  Human detection pipeline; usage and tuning details are documented
  in the file header.

These scripts represent the **legacy AI path** and remain for validation and comparison.

---

### YOLO ONNX pipelines (recommended direction)

YOLO ONNX pipelines represent the current and future direction of the project.
They emphasize stronger event semantics, clearer evidence artifacts,
and more explicit logging behavior.

- `_14.5_AICAM_Drones_YoloOnnx.py`  
  YOLO ONNX drone detection pipeline with:
  - confirmation windows
  - rotating operational logs
  - annotated and raw clip output

- `_15.2_AICAM_Intrusion.py`  
  “IntrusionCam” pipeline using YOLO ONNX.
  This script is intentionally minimal and keeps key paths hard-coded near the top
  to make system behavior explicit during planning and testing.

---

### Supporting documentation

Architecture and operational guides:

- `_ARCHITECTURE.md`
- `_CAMERA_SETTINGS_GUIDE.md`
- `_CAMERA_ENCRYPTION_OUTBOX_GUIDE*.md`
- `_SECURE_CAMERA_RECEIVER_SYNC.md`
- `_RECEIVER_VERIFICATION_INGEST_GUIDE.md`

Training and tuning material:

- `_14.*` drone training documents
- `_DRONE_TUNING_Legacy.md`

---

## Critical constraint: one camera owner

Only **one** camera pipeline may run at any given time.

Violating this constraint typically results in:

- `pipeline handler in use by another process`
- intermittent capture failures
- libcamera resource contention

**Operational rule:**
1. Start one service or script
2. Allow it to fully own the camera
3. Stop it cleanly before starting another

This applies equally to `rpicam-*` tools and Picamera2 pipelines.

---

## Requirements (Raspberry Pi OS)

### Baseline camera scripts

- Raspberry Pi OS with the libcamera stack
- `rpicam-still` and `rpicam-vid` available in `PATH`

### Picamera2 + OpenCV pipelines

Install using apt (preferred for ABI compatibility on Raspberry Pi):

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv ffmpeg
```

Notes:
- Avoid mixing pip-installed OpenCV with apt-installed Picamera2.
- Headless operation is supported; preview modes require an active desktop session.

---

## Storage model (current behavior)

Most services follow this intent:

- **Preferred output**: external storage mounted at `/media/user/disk`
- **Fallback output**: a user-writable path under the home directory

Mount detection is performed at runtime.
If the external disk is not mounted, services continue operating using fallback paths.

Important notes:

- Some scripts intentionally create directories only after confirming mount state.
- Other scripts predate this rule and may be updated later.
- Placeholder paths (e.g. `/home/usr/...`) reflect in-progress normalization.

This variation is intentional during the planning and stabilization phase.

---

## Logging model (current behavior)

Logging is deliberately split by purpose:

### Operational logs
- Runtime status, warnings, debug output
- May rotate
- May be written to files or system logs depending on the script

### Event logs
- Append-only records of semantic events
- Examples: `HUMAN_ENTER`, `HUMAN_EXIT`
- Often written as:
  - `events.log` (human-readable)
  - `events.jsonl` (machine-parseable)

Event logs are treated as **evidence artifacts**, not transient debug output.

Logging consolidation is planned but not yet enforced across all services.

---

## Quick start (validation)

```bash
python3 1_rpiCAMStill.py
python3 2_rpiVIDTest.py
python3 3_rpiVIDSec.py
```

YOLO pipelines include extensive CLI flags; consult file headers before running.

---

## Project status

This repository reflects an **active planning and validation phase**.

- Behavior is prioritized over uniformity
- Duplication is tolerated to surface hidden assumptions
- Consolidation will occur only after storage, logging, and failure policies are finalized

Differences between scripts document the evolution of the system rather than mistakes.

---

## Planned consolidation targets

When policy decisions are finalized, the following will be addressed:

1. Single authoritative storage path selection logic
2. Unified logging configuration (paths, rotation, verbosity)
3. Consistent directory layout for all artifacts
4. Systemd units with explicit mount dependencies and sandboxing. 

## Python Support

In addition to shell-based operation, this project supports Python-based workflows.
Python usage is intended to run inside an isolated virtual environment (venv) to
preserve dependency integrity and reduce host impact. Operational details, including
example commands and environment setup, are documented in `OPERATIONS.md`.

Refer to **OPERATIONS.md** for authoritative, step-by-step execution guidance.

![image](https://github.com/user-attachments/assets/dcc26235-be04-41bb-84ae-3902600018cd)

![image](https://github.com/user-attachments/assets/c313497d-826e-4782-a1a4-129193c0e56d)

![image](https://github.com/user-attachments/assets/fcc818ff-463b-41d2-97ed-3719990adcc6)
