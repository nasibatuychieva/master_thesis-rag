

import json
from pathlib import Path

# --------------------------------------------------
# Paths (Windows)
# --------------------------------------------------
INPUT_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset.jsonl"
)

OUTPUT_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Filter only "summary"
# --------------------------------------------------
kept = 0
skipped = 0

with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            print(f"⚠️ JSON error in line {line_no}, skipped.")
            continue

        if record.get("query_type") == "summary":
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
        else:
            skipped += 1

print(f"✅ Done. Kept {kept} summary questions. Skipped {skipped} non-summary (or invalid) lines.")
print(f"📄 Output: {OUTPUT_PATH}")
