import pandas as pd

turns = pd.read_csv('data/processed/apple_support_resolved_turns.csv')
t0 = turns[turns['turn_index'] == 0]

patterns = [
    'say ', 'ignore ', 'pretend', 'developer mode', 'jailbreak', 
    'refund approved', 'tell them', 'system prompt', 'repeat after',
    'you must say', 'respond with', 'apple will replace'
]

print(f"Total turn 0 tweets in resolved dataset: {len(t0)}")
for p in patterns:
    matches = t0[t0['text'].str.lower().str.contains(p, na=False)]
    print(f"Pattern '{p}': {len(matches)} matches")
    for idx, row in matches.head(3).iterrows():
        clean_msg = row['text'].replace('\n', ' ')
        print(f"   [{row['case_id']}] {clean_msg[:120]}")
