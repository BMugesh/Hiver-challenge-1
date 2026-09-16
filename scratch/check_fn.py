import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.security_detector import detect_security_intent

with open("data/security/prompt_injection/prompt_injection_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["label"] == "INJECTION":
            res = detect_security_intent(item["text"])
            if not res.is_prompt_injection:
                print(f"[{item['attack_type']}]: {item['text']}")
