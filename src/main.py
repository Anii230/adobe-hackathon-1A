import os

import json
from heading_detection.stateful_outline import generate_outline

INPUT_PDF = "data/input/sample.pdf"
OUTPUT_JSON = "data/output/sample_output.json"

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
result = generate_outline(INPUT_PDF)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"[✓] Output written to {OUTPUT_JSON}")
