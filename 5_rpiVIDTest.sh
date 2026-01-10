#!/usr/bin/env bash
set -euo pipefail
VIDEO_DIR="/media/user/disk/videos"
FALLBACK_DIR="/home/user/videos"
LOG_FILE="/var/log/rpicam/rpicam.log"
MOUNTPOINT="/media/user/disk"
sudo mkdir -p "$(dirname "$LOG_FILE")" || true
mkdir -p "$VIDEO_DIR" "$FALLBACK_DIR"
session_ts="$(date +%Y%m%d_%H%M%S)"
if mountpoint -q "$MOUNTPOINT"; then
  out_dir="$VIDEO_DIR"
else
  out_dir="$FALLBACK_DIR"
fi
out_file="${out_dir}/videos_${session_ts}.mp4"
{
  printf "\n=== Vid-Test Started: %s ===\n" "$session_ts"
  printf "OUT: %s\n" "$out_file"
  rpicam-vid --timeout "6000" --nopreview --codec h264 -o "$out_file"
  end_ts="$(date +%Y%m%d_%H%M%S)"
  printf "=== Vid-Test ended (session %s, ended %s) ===\n" "$session_ts" "$end_ts"
} >>"$LOG_FILE" 2>&1
printf "%s\n" "$out_file"
