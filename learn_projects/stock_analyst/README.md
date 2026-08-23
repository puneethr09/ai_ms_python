# 📈 Project 6: "Night Shift & Day Shift" Indian Stock Fundamental Analyst (Pi 5)

> **Autonomous Asynchronous Equity Research Engine on Raspberry Pi 5**

---

## 📌 Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│ 🌙 THE NIGHT SHIFT (night_shift.py - 2:00 AM Cron)     │
│    - Fetches NSE/BSE stocks (RELIANCE, TCS, INFY, etc) │
│    - Evaluates valuation via Local LLM on Port 8080    │
│    - Writes structured intelligence into SQLite db     │
├────────────────────────────────────────────────────────┤
│ ☀️ THE DAY SHIFT (day_shift.py - Market Hours)         │
│    - Instant 0.01s retrieval of precomputed valuations │
│    - Zero perceived waiting time, zero daytime CPU tax │
└────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup & Installation on Raspberry Pi 5

```bash
cd /home/puneeth/repo/ai_ms_python/learn_projects/stock_analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Running the Night Shift (Batch Analysis)

Make sure `llama-server` is running on port 8080:
```bash
python night_shift.py
```

---

## ⚡ Instant Daytime Querying

```bash
# View morning dashboard of all tracked stocks:
python day_shift.py

# Instant deep-dive valuation for a specific stock:
python day_shift.py TATAMOTORS
python day_shift.py TCS
python day_shift.py RELIANCE
```
