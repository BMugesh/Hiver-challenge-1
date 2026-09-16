"""
SupportDNA Master Knowledge Layer Orchestrator
==============================================
Runs end-to-end knowledge extraction across all 4 layers, executes the zero-leakage
audit, and compiles statistical reports.

Usage:
    python src/knowledge/run_knowledge_layers.py
"""

import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.build_intent_language import build_intent_language_layer
from src.knowledge.build_resolution_knowledge import build_resolution_knowledge_layer
from src.knowledge.build_escalation_knowledge import build_escalation_knowledge_layer
from src.knowledge.build_safety_knowledge import build_safety_knowledge_layer
from src.knowledge.leakage_check import run_leakage_check
from src.knowledge.statistics import generate_statistics_reports


def run_all():
    print("=" * 80)
    print("SUPPORTDNA KNOWLEDGE LAYER GENERATION PIPELINE")
    print("=" * 80)
    t0 = time.time()

    print("\n>>> STEP 1/6: Building Layer 1 — Intent & Language Knowledge...")
    t1 = time.time()
    l1_stats = build_intent_language_layer()
    print(f"Layer 1 completed in {time.time() - t1:.2f}s.")

    print("\n>>> STEP 2/6: Building Layer 2 — Resolution Knowledge...")
    t2 = time.time()
    l2_stats = build_resolution_knowledge_layer()
    print(f"Layer 2 completed in {time.time() - t2:.2f}s.")

    print("\n>>> STEP 3/6: Building Layer 3 — Escalation Knowledge...")
    t3 = time.time()
    l3_stats = build_escalation_knowledge_layer()
    print(f"Layer 3 completed in {time.time() - t3:.2f}s.")

    print("\n>>> STEP 4/6: Building Layer 4 — Safety & Edge-Case Knowledge...")
    t4 = time.time()
    l4_stats = build_safety_knowledge_layer()
    print(f"Layer 4 completed in {time.time() - t4:.2f}s.")

    print("\n>>> STEP 5/6: Running Strict Zero-Leakage Audit...")
    t5 = time.time()
    leakage_res = run_leakage_check()
    print(f"Leakage Audit completed in {time.time() - t5:.2f}s with status: {leakage_res['status']}.")

    print("\n>>> STEP 6/6: Generating Statistics and Utilization Reports...")
    t6 = time.time()
    stat_reports = generate_statistics_reports()
    print(f"Statistics generated in {time.time() - t6:.2f}s.")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL 4 KNOWLEDGE LAYERS GENERATED SUCCESSFULLY IN {total_time:.2f}s")
    print(f"Leakage Status: {leakage_res['status']}")
    print(f"Reports: reports/knowledge_layer_statistics.txt, reports/data_utilization_before_after.txt")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
