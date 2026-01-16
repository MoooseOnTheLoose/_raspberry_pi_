# 🔐 Secure Camera → Receiver Sync
### SSH-First • Encrypted • Offline-Safe Transport (Project Default)

## 🔗 This guide pairs with

- **CAMERA_ENCRYPTION_OUTBOX_GUIDE.md** or **CAMERA_ENCRYPTION_OUTBOX_GUIDE_GNUPG.md** — camera-side encryption & OUTBOX
- **SECURE_CAMERA_RECEIVER_SYNC.md** — SSH transport & automation
- **RECEIVER_VERIFICATION_INGEST_GUIDE.md** — receiver-side verification & ingest

These guides form a single, coherent pipeline and are intended to be used together.


> **Rule:** 🔒 Encrypt first → 🚚 transport second → ✅ verify always  
> This document covers **transport + automation**. Encryption is handled by the camera-side guide(s).

---

## 📌 Table of contents
- [What this is](#-what-this-is)
- [Architecture](#-architecture)
- [Directory layout](#-directory-layout)
- [Receiver setup](#-receiver-setup)
- [Camera setup](#-camera-setup)
- [Manual transfer test](#-manual-transfer-test)
- [Automation with systemd timer](#-automation-with-systemd-timer)
- [Offline fallback](#-offline-fallback)
- [Hardening notes](#-hardening-notes)
- [Anti-patterns](#-anti-patterns)

---

## 🧠 What this is

This guide sets up a reliable, resumable **SSH transfer** from a camera node to a receiver node.

🟢 Works on:
- Raspberry Pi OS (64-bit) / Debian (systemd)
- Internet, private LAN, isolated switch, or direct Ethernet cable

🔐 Security assumption:
- The camera **only ships encrypted artifacts** (e.g., `.gpg` or `.age`) from an **OUTBOX** directory.
- The receiver stores ciphertext only.

---

## 🏗️ Architecture

```
Camera node (trusted for plaintext)
  ├─ records clips to:  videos/   (plaintext, local only)
  ├─ encrypts to:       outbox/   (ciphertext only)
  └─ ships outbox/  ───────────▶  Receiver node (ingest vault)
                                  └─ stores ciphertext + hashes + manifests
```

---

## 📁 Directory layout

### Camera
```
/media/user/disk/videos/     # plaintext (never synced)
/media/user/disk/outbox/     # encrypted artifacts (sync source)
```

### Receiver
```
/srv/cam_ingest/cam1/        # encrypted archive (sync destination)
```

---

## 🖥️ Receiver setup

### 1) Create an ingest user + destination directory
```bash
sudo adduser --disabled-password --gecos "" ingest
sudo mkdir -p /srv/cam_ingest/cam1
sudo chown -R ingest:ingest /srv/cam_ingest
sudo chmod 750 /srv/cam_ingest /srv/cam_ingest/cam1
```

### 2) Lock down SSH for ingest user (recommended baseline)
Create a drop-in:
```bash
sudo nano /etc/ssh/sshd_config.d/ingest.conf
```

Add:
```text
Match User ingest
  PasswordAuthentication no
  PubkeyAuthentication yes
  X11Forwarding no
  AllowTcpForwarding no
  PermitTTY no
```

Reload:
```bash
sudo systemctl reload ssh
```

### 3) Firewall (recommended)
Allow SSH **only** from the camera IP (example UFW):
```bash
sudo ufw allow from <CAMERA_IP> to any port 22 proto tcp
```

---

## 📷 Camera setup

### 1) Generate a dedicated SSH key for ingest
```bash
ssh-keygen -t ed25519 -f ~/.ssh/cam_ingest_ed25519 -N ""
```

### 2) Install the public key on the receiver ingest user
```bash
ssh-copy-id -i ~/.ssh/cam_ingest_ed25519.pub ingest@<RECEIVER_IP>
```

### 3) Quick connectivity check
```bash
ssh -i ~/.ssh/cam_ingest_ed25519 -o BatchMode=yes ingest@<RECEIVER_IP> 'echo OK'
```

---

## 🧪 Manual transfer test

This command ships **only encrypted OUTBOX contents** and removes local copies only after success:

```bash
rsync -av --partial --inplace --remove-source-files   -e "ssh -i ~/.ssh/cam_ingest_ed25519 -o BatchMode=yes -o ConnectTimeout=10"   /media/user/disk/outbox/   ingest@<RECEIVER_IP>:/srv/cam_ingest/cam1/
```

Notes:
- `--partial --inplace` helps resume interrupted transfers.
- `--remove-source-files` makes OUTBOX behave like a queue.
- Keep OUTBOX ciphertext-only (`.gpg` / `.age` + `.sha256` + manifests).

---

## ⏱️ Automation with systemd timer (preferred)

### 1) Create a service unit (camera)
`/etc/systemd/system/cam-sync.service`
```ini
[Unit]
Description=Sync encrypted OUTBOX to receiver
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=user
ExecStart=/usr/bin/rsync -av --partial --inplace --remove-source-files   -e "ssh -i /home/user/.ssh/cam_ingest_ed25519 -o BatchMode=yes -o ConnectTimeout=10"   /media/user/disk/outbox/   ingest@<RECEIVER_IP>:/srv/cam_ingest/cam1/
```

Replace:
- `User=user` with your camera username
- `/home/user/...` with the correct home directory
- `<RECEIVER_IP>` with receiver address

### 2) Create a timer unit (camera)
`/etc/systemd/system/cam-sync.timer`
```ini
[Unit]
Description=Run cam-sync every 2 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target
```

### 3) Enable
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cam-sync.timer
systemctl list-timers | grep cam-sync
```

### 4) Logs / debugging
```bash
journalctl -u cam-sync.service -n 200 --no-pager
```

---

## 📴 Offline fallback

If there is **no IP path at all**:
- Copy `/media/user/disk/outbox/` to removable media (USB SSD)
- Move/copy onto the receiver into `/srv/cam_ingest/cam1/`
- Run verification on receiver (see receiver verification guide)

This is the **same pipeline** with a different transport.

---

## 🔒 Hardening notes (recommended)

🟦 Receiver:
- Keep services minimal (SSH only)
- Use a dedicated ingest disk/mount if available
- Consider periodic hash audits

🟦 Camera:
- OUTBOX must contain ciphertext only
- Treat transfer failures as “retry later,” not “send plaintext”

---

## ❌ Anti-patterns

🚫 Syncing `videos/` (plaintext directory)  
🚫 Relying on SSH/VPN alone as “encryption”  
🚫 Remote LUKS unlock for replication  
🚫 NFS/SMB network mounts for ingest  
🚫 Auto-decrypt on receiver

---

**End of document**


## 📁 Standard Directory Layout (Project-Wide)

Unless explicitly stated otherwise, all guides use the following layout for **encrypted ingest data**:

```
<BASE_PATH>/
├── encrypted/     # encrypted artifacts (.gpg / .age)
├── hashes/        # integrity hashes (.sha256)
├── manifests/     # JSON manifests
└── quarantine/    # failed or unverified files
```

Notes:
- Plaintext is **never** stored here
- Only `encrypted/` is transported between systems
- `quarantine/` is for investigation only and is never synced
