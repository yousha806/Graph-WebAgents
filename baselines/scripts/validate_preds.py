import sys
import json
from typing import List

REQUIRED_KEYS = [
    "gt_element", "gt_action", "gt_value",
    "pred_element", "pred_action", "pred_value",
    "candidates"
]


def validate_preds(path: str) -> List[str]:
    errors = []
    count = 0
    with open(path, "r", encoding="utf8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                errors.append(f"Line {i}: empty")
                continue
            try:
                j = json.loads(line)
            except Exception as e:
                errors.append(f"Line {i}: JSON parse error: {e}")
                continue
            count += 1
            for k in REQUIRED_KEYS:
                if k not in j:
                    errors.append(f"Line {i}: missing key '{k}'")
            # candidates type
            if "candidates" in j and not isinstance(j["candidates"], list):
                errors.append(f"Line {i}: 'candidates' is not a list")
            # value_states shape
            if "value_states" in j and j["value_states"] is not None:
                if not isinstance(j["value_states"], list):
                    errors.append(f"Line {i}: 'value_states' is not a list")
                else:
                    # check numeric
                    for v in j["value_states"]:
                        if not isinstance(v, (int, float)):
                            errors.append(f"Line {i}: 'value_states' contains non-numeric element")
                            break
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_preds.py <preds.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    errs = validate_preds(path)
    if not errs:
        print(f"Validation passed: {path} looks good.")
    else:
        print(f"Validation errors ({len(errs)}):")
        for e in errs[:200]:
            print(" - ", e)
        if len(errs) > 200:
            print(f"... and {len(errs)-200} more errors")
