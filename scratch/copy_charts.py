"""
Copy report PNGs to artifact directory for proper embedding.
"""
import shutil
from pathlib import Path

source_dir = Path(r"c:\Hiver\reports")
target_dir = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\b41e092e-4819-4b5d-9e00-7ed30bf70de4")

for name in ["stage8_confusion_matrix.png", "stage8_intent_metrics.png", "stage8_retrieval_metrics.png", "stage8_decision_metrics.png"]:
    src = source_dir / name
    dst = target_dir / name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {name} to {dst}")
