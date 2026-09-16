"""
SupportDNA Knowledge Layer Package
==================================
Modular 4-layer knowledge extraction and data utilization system for customer support.
"""

from src.knowledge.provenance import ProvenanceRecord
from src.knowledge.build_intent_language import build_intent_language_layer
from src.knowledge.build_resolution_knowledge import build_resolution_knowledge_layer
from src.knowledge.build_escalation_knowledge import build_escalation_knowledge_layer
from src.knowledge.build_safety_knowledge import build_safety_knowledge_layer
from src.knowledge.leakage_check import run_leakage_check
from src.knowledge.statistics import generate_statistics_reports

__all__ = [
    "ProvenanceRecord",
    "build_intent_language_layer",
    "build_resolution_knowledge_layer",
    "build_escalation_knowledge_layer",
    "build_safety_knowledge_layer",
    "run_leakage_check",
    "generate_statistics_reports"
]
