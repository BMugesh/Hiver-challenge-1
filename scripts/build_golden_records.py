"""
SupportDNA Golden Evaluation Set — Master Record Assembler
==========================================================
Combines Part 1 (GOLDEN_0001 - GOLDEN_0098) and Part 2 (GOLDEN_0099 - GOLDEN_0200)
into the unified 200 hand-labelled evaluation set.
"""

from typing import List, Dict, Any, Set
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.golden_records_part1 import get_part_1_records
from scripts.golden_records_part2 import get_part_2_records


def generate_golden_records(
    candidates_df: pd.DataFrame,
    excluded_roots: Set[int],
    excluded_texts: Set[str]
) -> List[Dict[str, Any]]:
    """
    Assemble the complete 200 Golden Evaluation records.
    Verifies that no record overlaps with excluded split roots or split texts.
    """
    part1 = get_part_1_records()
    part2 = get_part_2_records()
    all_records = part1 + part2

    if len(all_records) != 200:
        raise ValueError(f"Expected 200 records, got {len(all_records)}")

    # Verify zero leakage
    import re
    for r in all_records:
        norm = r["customer_message"].strip().lower()
        if norm in excluded_texts:
            raise ValueError(f"LEAKAGE: Message present in split texts: {r['customer_message']}")
        
        # Check source_id thread root
        match = re.search(r'\d+', r["source_id"])
        if match:
            rid = int(match.group(0))
            if rid in excluded_roots:
                raise ValueError(f"LEAKAGE: Source thread root {rid} ({r['source_id']}) present in excluded split roots!")

    return all_records
