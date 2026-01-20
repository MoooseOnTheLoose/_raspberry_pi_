🛠️ **OPERATIONS.md — Operations Guide**
Rasp AI Camera (AICAM / Rasp)

UPDATE:1/20/2026
---

## Why This Document Exists

This document defines the **correct, repeatable, and secure way to operate the system**.

If ARCHITECTURE.md explains *what* the system is and *why* it exists,  
OPERATIONS.md explains **how it is meant to be used without breaking its guarantees**.

This is not optional reading for operators.

---

## Operational Philosophy (Read First)

The system is designed around four operational truths:

1. **Nothing is automatic unless explicitly intended**
2. **Storage is ephemeral, not permanent**
3. **Failure must stop progress**
4. **Operator intent is a security boundary**

If you try to “make it easier” without understanding these, you will weaken the system.

---

## Preconditions

Before operation, ensure:

- You have physical access to the device
- You have the disk passphrase
- The system is booted to a known-good state
- You understand where output data will be written

Do not operate this system remotely unless explicitly documented and secured.

---

## Boot Procedure

1. Power on the Raspberry Pi
2. Allow OS to fully boot
3. Log in locally (console or secure session)

Verify system state:
```bash
uptime
lsblk
```

Confirm the encrypted disk is **not** unlocked yet.

---

## Encrypted Storage Lifecycle (Critical)

### Unlock Disk

```bash
sudo cryptsetup open /dev/sda crypt-videos
```

Confirm:
```bash
ls /dev/mapper/crypt-videos
```

If this fails, **stop**. Do not continue.

---

### Mount Filesystem

```bash
sudo mount /dev/mapper/crypt-videos /media/user/disk
```

Verify:
```bash
mount | grep crypt-videos
```

Confirm directories exist:
```bash
ls /media/user/disk/images
ls /media/user/disk/videos
```

If directories are missing, **stop and investigate**.

---

## Camera Operation

### Still Image Capture

Example:
```bash
./rpicam-still.sh
```

Expected behavior:
- Script exits on error
- Output written only to encrypted storage
- No background processes remain

Verify output:
```bash
ls -lh /media/user/disk/images
```

---

### Video Capture

Example:
```bash
./rpicam-video.sh
```

Verify:
```bash
ls -lh /media/user/disk/videos
```

Do not assume success without verification.

---

## Failure Handling Rules

If **any** of the following occur:
- Script exits with non-zero code
- Storage becomes unavailable
- Disk fills unexpectedly
- Camera fails to initialize

Then:
1. Stop execution
2. Preserve current state
3. Do **not** retry blindly
4. Investigate logs or script output

Silent retries are prohibited.

---

## Post-Capture Shutdown Procedure

### Unmount Filesystem

```bash
sudo umount /media/user/disk
```

Verify:
```bash
mount | grep crypt-videos
```
(No output is expected.)

---

### Close Encrypted Container

```bash
sudo cryptsetup close crypt-videos
```

Verify:
```bash
ls /dev/mapper | grep crypt-videos
```
(No output is expected.)

---

## Power Down

Only after storage is closed:

```bash
sudo poweroff
```

Never power off with mounted encrypted storage.

---

## Safe Defaults Checklist

Before walking away:
- [ ] Disk is unmounted
- [ ] Encrypted container is closed
- [ ] No capture scripts are running
- [ ] System is powered down or idle

---

## Common Mistakes (Avoid These)

❌ Leaving disk mounted “just in case”  
❌ Running capture scripts before mounting storage  
❌ Assuming output exists without checking  
❌ Power loss with mounted encrypted disk  
❌ Adding automation without documentation  

---

## What This Document Protects Against

- Accidental plaintext writes
- Data loss due to improper shutdown
- Silent failures
- Operator confusion
- Security drift over time

---

## When to Update This File

Update OPERATIONS.md if:
- Disk device names change
- Mount paths change
- Scripts change behavior
- Automation is introduced (opt-in only)

If the system changes and this file does not, **the documentation is wrong**.

---

## Final Rule

> **If an operation is not written here, it is not part of the supported workflow.**
