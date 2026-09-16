import json
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.security_detector import SecurityDetector

# Test on validation set
val_path = PROJECT_ROOT / "data" / "security" / "prompt_injection" / "prompt_injection_validation.jsonl"
with open(val_path, "r", encoding="utf-8") as f:
    val_data = [json.loads(line) for line in f]

det = SecurityDetector()
val_correct = 0
for item in val_data:
    res = det.analyze(item["text"])
    expected = (item["label"] == "INJECTION")
    if res.is_prompt_injection == expected:
        val_correct += 1
    else:
        print(f"Val mismatch: '{item['text']}', expected={expected}, got={res.is_prompt_injection}")

print(f"Validation Accuracy: {val_correct}/{len(val_data)} ({val_correct/len(val_data)*100:.1f}%)")
