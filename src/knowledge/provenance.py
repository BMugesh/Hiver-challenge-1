"""
SupportDNA Knowledge Layer — Data Provenance
===========================================
Ensures full auditability and traceability for every derived record across
all 4 knowledge layers back to original tweets, conversation threads, and splits.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ProvenanceRecord:
    """
    Standardized provenance metadata attached to all derived knowledge records.
    Ensures zero anonymous records and preserves split boundaries.
    """
    source_thread_id: int
    source_message_id: Optional[int] = None
    source_case_id: Optional[str] = None
    source_dataset: str = "apple_support_threads.json"
    original_status: str = "UNKNOWN"
    derived_from_stage: str = "STAGE_2_RECONSTRUCTED"
    split: str = "unassigned_excluded"  # train, validation, test, unassigned_excluded

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance object to clean dictionary."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
