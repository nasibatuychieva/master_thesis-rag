# import json
# import os
# from pathlib import Path

# # --------------------------------------------------
# # 1) Project root & paths
# # --------------------------------------------------

# def _guess_project_root() -> Path:
#     return Path(__file__).resolve().parents[2]  # ✅ korrekt

# PROJECT_ROOT = _guess_project_root()

# INPUT_PATH = Path(
#     os.getenv(
#         "ANSWERS_INPUT_PATH",
#         PROJECT_ROOT / "main" / "evaluation" / "graphrag" / "answers_log_new_dataset.jsonl"
#     )
# ).expanduser().resolve()

# OUTPUT_PATH = Path(
#     os.getenv(
#         "ANSWERS_OUTPUT_PATH",
#         PROJECT_ROOT / "main" / "evaluation" / "graphrag" / "answers_log_new_dataset_trimmed.jsonl"
#     )
# ).expanduser().resolve()

# # --------------------------------------------------
# # 2) Trim logic
# # --------------------------------------------------

# FIELDS_TO_KEEP = {
#     "script",          
#     "question_id",
#     "query_type",
#     "question",
#     "answer",
#     "gold_answer",
# }

# OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# with INPUT_PATH.open("r", encoding="utf-8") as fin, \
#      OUTPUT_PATH.open("w", encoding="utf-8") as fout:

#     for line in fin:
#         if not line.strip():
#             continue

#         record = json.loads(line)

#         trimmed = {k: record.get(k) for k in FIELDS_TO_KEEP}

#         fout.write(json.dumps(trimmed, ensure_ascii=False) + "\n")

# print(f"Trimmed dataset written to: {OUTPUT_PATH}")

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
