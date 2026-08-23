import json
import logging
import httpx
from config import LLAMA_SERVER_URL, MODEL_NAME

logger = logging.getLogger("edge_sentinel.diagnostician")

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) running on a Raspberry Pi 5.
Analyze the provided Docker container crash log snippet.
Provide a concise, actionable diagnostic.

You MUST respond strictly in the following JSON format:
{
  "root_cause": "One clear sentence explaining what failed.",
  "severity": "HIGH | MEDIUM | LOW",
  "recommended_fix": "Single bash command to resolve the issue (e.g. docker restart container_name)."
}
"""

def diagnose_error(container_name: str, log_snippet: str) -> dict:
    """
    Sends the raw Docker error trace to the local llama-server on Pi 5
    and returns a structured diagnosis dictionary.
    """
    # Keep only the last 15 lines so ARM CPU prompt prefill takes < 1 second!
    clean_lines = [line for line in log_snippet.strip().split("\n") if line.strip()]
    trimmed_log = "\n".join(clean_lines[-15:])

    user_prompt = f"Container: {container_name}\n\nLog Trace:\n```\n{trimmed_log}\n```"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # Low temperature for deterministic, precise diagnostics
        "max_tokens": 128
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(LLAMA_SERVER_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"].strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()

            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                return {
                    "root_cause": raw_content,
                    "severity": "HIGH",
                    "recommended_fix": f"docker restart {container_name}"
                }

    except Exception as e:
        logger.error(f"Failed to communicate with local AI server at {LLAMA_SERVER_URL}: {e}")
        return {
            "root_cause": f"Error detected in {container_name}, but local AI diagnostician was unavailable ({e}).",
            "severity": "HIGH",
            "recommended_fix": f"docker restart {container_name}"
        }
