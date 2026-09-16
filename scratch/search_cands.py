import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scripts.build_golden_set import load_exclusion_sets, get_held_out_candidates

roots, cases, texts = load_exclusion_sets()
candidates = get_held_out_candidates(roots, texts)

# find candidates talking about purchases / billing / itunes / app store
billing_cands = candidates[candidates['customer_text'].str.contains(r'purchase|subscription|refund|charged|receipt|billing', case=False, na=False)]
print('Billing candidates:', len(billing_cands))
for i, r in billing_cands.head(15).iterrows():
    root_id = int(r['thread_root_id'])
    if root_id not in roots:
        print(f"SRC_ROOT_{root_id}: {r['customer_text']}")
