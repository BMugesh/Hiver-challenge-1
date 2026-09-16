import json
import jsonschema

with open('data/evaluation/golden_set_schema.json', 'r', encoding='utf-8') as f:
    schema = json.load(f)

with open('data/evaluation/golden_set.json', 'r', encoding='utf-8') as f:
    records = json.load(f)

print(f"Validating {len(records)} records...")
errors = []
for i, r in enumerate(records):
    try:
        jsonschema.validate(instance=r, schema=schema['items'])
    except jsonschema.ValidationError as e:
        errors.append((r.get('golden_id'), list(e.path), e.message))

print(f"Total errors: {len(errors)}")
for gid, path, msg in errors:
    print(f"{gid}: path={path}, msg={msg}")
