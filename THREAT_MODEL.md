🎯 **THREAT_MODEL.md — Threat Model**
Rasp AI Camera (AICAM / Rasp)

UPDATE:1/21/2026
---

## Attack Surfaces
### Physical

- USB storage removal
- Power cycling
- Direct console access

**Residual Risk**
- Hardware tampering is possible but out of scope

---

### Software

- Camera capture scripts
- Shell environment
- Python helpers

**Mitigations**
- Minimal code paths
- AppArmor confinement
- Fail-fast execution

---

### Network

- Any enabled interface
- Outbound connections

**Mitigations**
- UFW default deny
- Explicit allow rules only

---

## Python Execution Surface
In addition to shell-based operation, the system may be operated via Python scripts.
Python execution is constrained to local use within a dedicated virtual environment
(venv). The interpreter, standard library, and installed dependencies are considered
part of the trusted computing base.

The threat model assumes:
- No implicit or automatic network access by Python code
- Dependencies are installed intentionally and verified by the operator
- Python is used only to orchestrate capture, validation, and storage operations,
  not to expand system connectivity or exposure

This subsection acknowledges Python as an execution surface without expanding the
system’s threat assumptions beyond existing local code execution risks.

## Why This Document Exists
This document makes explicit the **threat assumptions already embedded** in the architecture and operations.

Threat modeling here serves three purposes:
1. Educate the reader on realistic risks
2. Prevent accidental scope creep
3. Anchor future decisions to explicit security goals

This is not paranoia.  
This is **bounded realism**.

---

## Methodology
This threat model loosely follows **STRIDE-style thinking**, but is adapted for a **single-purpose, embedded, offline-first system**.

We model:
- Assets
- Adversaries
- Attack surfaces
- Mitigations
- Explicitly out-of-scope threats

---

## Assets (What We Are Protecting)
### Primary Assets (High Value)

1. **Captured Images and Videos**
   - Confidentiality is critical
   - Integrity is important
   - Availability is secondary

2. **Metadata**
   - Timestamps
   - Filenames
   - Capture cadence
   Metadata can be as sensitive as content.

3. **Disk Encryption Keys**
   - Passphrase material
   - Derived keys in memory (ephemeral)

---

### Secondary Assets (Medium Value)

- System configuration
- Capture scripts
- Firewall and MAC policy
- Operational procedures

---

## Adversary Classes
### 1. Opportunistic Thief

**Capabilities**
- Physical access
- Can steal device or disk
- No specialized tooling

**Goals**
- Extract stored data
- Resell hardware

**Mitigations**
- Full-disk encryption (LUKS2)
- No plaintext data at rest when closed

---

### 2. Curious Insider

**Capabilities**
- Authorized access at some point
- Familiar with Linux basics

**Goals**
- Browse captured data
- Explore system contents

**Mitigations**
- Manual disk lifecycle
- Explicit operational procedures
- Principle of least privilege

---

### 3. Network Observer / Opportunist

**Capabilities**
- Can observe or interact on the network
- Can scan ports and services

**Goals**
- Exfiltrate data
- Trigger unintended behavior

**Mitigations**
- Default-deny firewall
- No always-on services
- Outbound restrictions

---

### 4. Accidental Operator (Most Common)

**Capabilities**
- Legitimate access
- No malicious intent

**Goals**
- “Make it work”
- Speed and convenience

**Mitigations**
- OPERATIONS.md
- Fail-fast scripts
- Explicit workflows
- Loud failures

---

## STRIDE Mapping (Simplified)
| Category | Relevant | Notes |
|-------|----------|------|
| Spoofing | Low | No identity-based services |
| Tampering | Medium | Mitigated by manual ops |
| Repudiation | Low | Single-operator system |
| Information Disclosure | High | Primary threat |
| Denial of Service | Medium | Accepted risk |
| Elevation of Privilege | Medium | Mitigated by MAC |

---

## In-Scope Threats
✔ Disk theft  
✔ Accidental misoperation  
✔ Network exposure  
✔ Script compromise  
✔ Data exfiltration  

---

## Explicitly Out-of-Scope Threats
❌ Hardware implants  
❌ Side-channel attacks  
❌ Nation-state adversaries  
❌ Live forensic memory attacks  
❌ Supply-chain compromise  

These require fundamentally different architectures.

---

## Residual Risk (Accepted)
- Data loss due to operator error
- Device destruction
- Power loss during capture

These risks are accepted in favor of simplicity and clarity.

---

## Security Posture Summary
This system prioritizes:
1. Confidentiality
2. Predictability
3. Auditability

Over:
- Convenience
- Availability
- Automation

---

## When to Update This File
Update THREAT_MODEL.md if:
- New network capabilities are added
- Automation is introduced
- Remote access is enabled
- Trust assumptions change

If the threat model is outdated, **the security claims are invalid**.

---

## Final Principle
> **If a threat is not written here, it is either mitigated elsewhere or intentionally ignored.**
>
![image](https://github.com/user-attachments/assets/425668f0-dd15-4507-92a7-9cf54072e6a2)

