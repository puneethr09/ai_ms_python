import logging
import httpx
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("edge_sentinel.dispatcher")

def dispatch_alert(container_name: str, diagnosis: dict, raw_log_sample: str):
    """
    Formats the diagnostic alert and sends it via Telegram (if configured)
    and logs it to the system console.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity = diagnosis.get("severity", "HIGH").upper()
    root_cause = diagnosis.get("root_cause", "Unknown failure")
    recommended_fix = diagnosis.get("recommended_fix", f"docker restart {container_name}")

    severity_emoji = "🚨" if severity == "HIGH" else ("⚠️" if severity == "MEDIUM" else "ℹ️")

    # 1. Format Terminal Console Output
    console_msg = (
        f"\n{'='*60}\n"
        f"{severity_emoji} [AI OVERWATCHER ALERT] Container '{container_name}' Stalled!\n"
        f"⏰ Time: {timestamp}\n"
        f"📊 Severity: {severity}\n"
        f"🔍 Root Cause: {root_cause}\n"
        f"🛠️ Recommended Fix: {recommended_fix}\n"
        f"{'='*60}\n"
    )
    print(console_msg, flush=True)

    # 2. Dispatch to Telegram (if token is configured)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        telegram_text = (
            f"{severity_emoji} *[OVERWATCHER ALERT]* Container `{container_name}`\n\n"
            f"⏰ *Time:* `{timestamp}`\n"
            f"📊 *Severity:* `{severity}`\n\n"
            f"🔍 *AI Root Cause:* \n_{root_cause}_\n\n"
            f"🛠️ *Quick Fix Command:*\n`{recommended_fix}`\n\n"
            f"📜 *Raw Log Snippet:*\n`{raw_log_sample[:300]}`"
        )

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(telegram_url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": telegram_text,
                    "parse_mode": "Markdown"
                })
                if res.status_code == 200:
                    logger.info("Alert successfully dispatched to Telegram.")
                else:
                    logger.error(f"Telegram dispatch failed: {res.text}")
        except Exception as e:
            logger.error(f"Failed to connect to Telegram API: {e}")
