import os
import re

# 1. Local AI Engine Configuration (llama-server on Pi 5)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Llama-3.2-1B-Instruct")

# 2. Monitored Docker Containers
MONITORED_CONTAINERS = [
    "immich_server",
    "immich_postgres",
    "immich_machine_learning",
    "immich_redis",
    "caloric",
    "plex",
    "radarr",
    "sonarr",
    "prowlarr",
    "deluge",
    "pihole",
    "scrutiny",
    "uptime-kuma",
    "stock_fundamental-stock-analysis-app-1"
]

# 3. Crash & Error Detection Patterns
ERROR_PATTERN = re.compile(
    r"(?i)\b(ERROR|FATAL|panic|Traceback|Exception|OOMKilled|ConnectionRefused|database is locked|SIGSEGV|Critical)\b"
)

# 4. Telegram Notification Settings (Optional - set via environment variables)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# 5. Cooldown (Seconds to suppress duplicate alerts for the same container)
ALERT_COOLDOWN_SECONDS = 300  # 5 minutes
