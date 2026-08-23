#!/usr/bin/env bash
set -e

# ==============================================================================
# 🌙 THE NIGHT SHIFT: 2:00 AM Autonomous Indian Stock Valuation Runner
# ==============================================================================

DIR="/home/puneeth/repo/ai_ms_python/learn_projects/stock_analyst"
cd "$DIR"

# Ensure venv exists
if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
fi

echo "========================================================" >> "$DIR/night_shift.log"
echo "🌙 [$(date '+%Y-%m-%d %H:%M:%S')] Starting Night Shift Batch Run..." >> "$DIR/night_shift.log"

python "$DIR/night_shift.py" --universe all >> "$DIR/night_shift.log" 2>&1

echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] Night Shift Batch Completed!" >> "$DIR/night_shift.log"
