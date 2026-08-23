import time
import logging
import argparse
import docker
from datetime import datetime
from config import (
    MONITORED_CONTAINERS,
    ERROR_PATTERN,
    ALERT_COOLDOWN_SECONDS
)
from diagnostician import diagnose_error
from dispatcher import dispatch_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("edge_sentinel")

# Cooldown map: { container_name: last_alert_timestamp }
last_alert_times = {}

def is_in_cooldown(container_name: str) -> bool:
    """Checks if an alert was recently dispatched for this container."""
    now = time.time()
    last_time = last_alert_times.get(container_name, 0)
    if (now - last_time) < ALERT_COOLDOWN_SECONDS:
        return True
    last_alert_times[container_name] = now
    return False

def handle_failure(container_name: str, log_snippet: str):
    """Diagnoses the failure using local LLM and dispatches alert."""
    if is_in_cooldown(container_name):
        logger.info(f"Suppressed duplicate alert for '{container_name}' (Cooldown active).")
        return

    logger.warning(f"🚨 Failure detected in '{container_name}'. Invoking Local AI Diagnostician...")
    
    # 1. Ask local Llama 3.2 1B on Pi 5 for diagnosis
    diagnosis = diagnose_error(container_name, log_snippet)
    
    # 2. Dispatch the formatted alert
    dispatch_alert(container_name, diagnosis, log_snippet)

def monitor_docker_events(client: docker.DockerClient):
    """
    Listens to the real-time Docker event stream for container death/OOM events.
    """
    logger.info("📡 Listening for Docker lifecycle events (die, oom, kill)...")
    
    for event in client.events(decode=True):
        if event.get("Type") == "container":
            action = event.get("Action", "")
            actor = event.get("Actor", {})
            container_name = actor.get("Attributes", {}).get("name", "")

            if container_name in MONITORED_CONTAINERS or not MONITORED_CONTAINERS:
                if action in ["die", "oom", "kill"]:
                    logger.error(f"Event triggered: Container '{container_name}' Action: {action}")
                    
                    try:
                        container = client.containers.get(container_name)
                        recent_logs = container.logs(tail=30).decode("utf-8", errors="replace")
                    except Exception:
                        recent_logs = f"Container {container_name} died with action: {action}"

                    handle_failure(container_name, recent_logs)

def scan_active_container_logs(client: docker.DockerClient):
    """
    Scans the last 20 lines of active containers for fatal exceptions or lock errors.
    """
    for name in MONITORED_CONTAINERS:
        try:
            container = client.containers.get(name)
            if container.status != "running":
                continue
            
            logs = container.logs(tail=20).decode("utf-8", errors="replace")
            if ERROR_PATTERN.search(logs):
                handle_failure(name, logs)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.debug(f"Could not inspect logs for {name}: {e}")

def run_simulation():
    """Runs a simulated error test to verify the local AI pipeline."""
    print("\n🔬 RUNNING SIMULATED CRASH TEST...")
    sample_container = "immich_postgres"
    sample_crash_log = (
        "2026-08-23 03:14:18.421 UTC [128] FATAL: remaining connection slots are reserved for non-replication superuser connections\n"
        "2026-08-23 03:14:18.422 UTC [129] ERROR: could not connect to server: Connection refused\n"
        "2026-08-23 03:14:18.423 UTC [130] PANIC: shared memory lock acquisition failed on /var/lib/postgresql/data\n"
    )
    print(f"Feeding simulated crash log to Local AI on Port 8080...\n")
    handle_failure(sample_container, sample_crash_log)

def main():
    parser = argparse.ArgumentParser(description="AI Overwatcher SRE Sentinel for Pi 5")
    parser.add_argument("--test", action="store_true", help="Run a simulated crash diagnostic test")
    args = parser.parse_args()

    if args.test:
        run_simulation()
        return

    print("=" * 60)
    print("🛡️  AI OVERWATCHER SRE SENTINEL DAEMON (Raspberry Pi 5)")
    print("=" * 60)

    try:
        client = docker.from_env()
        client.ping()
        logger.info("✅ Connected to Docker daemon via /var/run/docker.sock")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Docker daemon: {e}")
        print("\nTroubleshooting: Ensure your user is in the docker group ('sudo usermod -aG docker $USER')\n")
        return

    logger.info(f"Watching {len(MONITORED_CONTAINERS)} critical containers...")

    # Main monitoring loop
    while True:
        try:
            scan_active_container_logs(client)
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Sentinel shutting down cleanly.")
            break
        except Exception as e:
            logger.error(f"Unexpected loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
