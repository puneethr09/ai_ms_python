# 🛡️ Project 5: AI Overwatcher SRE Sentinel (Raspberry Pi 5)

> **24/7 Autonomous Edge Site Reliability Engineer powered by Local LLMs on ARM Cortex-A76**

---

## 📌 Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│ 1. THE WATCHMAN (sentinel.py)                          │
│    - Listens to /var/run/docker.sock                   │
│    - Monitors Plex, Immich, Radarr, Caloric, etc.      │
├────────────────────────────────────────────────────────┤
│ 2. THE LOCAL AI SRE (diagnostician.py)                 │
│    - Prompts local llama-server on Port 8080 (13 t/s) │
│    - Slices crash logs into: Root Cause + Quick Fix    │
├────────────────────────────────────────────────────────┤
│ 3. THE DISPATCHER (dispatcher.py)                      │
│    - Sends rich Telegram alerts to your phone          │
└────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup & Installation on Raspberry Pi 5

### 1. Create Virtual Environment & Install Dependencies
```bash
cd /home/puneeth/repo/ai_ms_python/learn_projects/edge_sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Quick Test: Simulated Crash Diagnostic

Make sure `llama-server` is running on port 8080:
```bash
python sentinel.py --test
```

---

## 🚀 Run as a 24/7 Background Systemd Service

```bash
sudo cp edge-sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edge-sentinel
sudo systemctl start edge-sentinel
```

Check status anytime:
```bash
sudo systemctl status edge-sentinel
```
