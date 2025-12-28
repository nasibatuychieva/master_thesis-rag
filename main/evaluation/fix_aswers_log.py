# import csv
# import ast
# import json
# from pathlib import Path

# # IN_PATH = Path(
# #     r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new.csv"
# # )
# IN_PATH = Path(
#     r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new_dataset.csv"
# )

# # OUT_PATH = IN_PATH.with_name("answers_log_fixed.csv")
# OUT_PATH = IN_PATH.with_name("answers_log_new_dataset_fixed.csv")

# EXPECTED_HEADER = [
#     "script",
#     "question_id",
#     "query_type",
#     "question",
#     "answer",
#     "gold_answer",
#     "n_context",
#     "context_preview",
#     "context_json",
# ]

# def looks_like_tuple_answer(s: str) -> bool:
#     s = (s or "").strip()
#     # heuristik: answer beginnt mit "(" und enthält typischerweise ein Tuple-Repr
#     return s.startswith("(") and s.endswith(")")

# def try_extract_tuple_answer(answer_field: str):
#     """
#     If answer_field looks like: "('text', [{'content':...}])"
#     returns (text, context_list) else (None, None)
#     """
#     try:
#         obj = ast.literal_eval(answer_field)
#         if isinstance(obj, tuple) and len(obj) == 2:
#             ans, ctx = obj
#             if isinstance(ans, str) and isinstance(ctx, (list, dict)):
#                 return ans, ctx
#     except Exception:
#         pass
#     return None, None

# def coerce_row_to_9_cols(row):
#     """
#     Force exactly 9 columns:
#     - if too many: merge extras into the last column (context_json)
#     - if too few: pad with ""
#     """
#     if len(row) == 9:
#         return row
#     if len(row) > 9:
#         head = row[:8]
#         tail = row[8:]
#         merged_last = ",".join(tail)
#         return head + [merged_last]
#     return row + [""] * (9 - len(row))

# def normalize_context_json(s: str) -> str:
#     """
#     Ensure context_json is valid JSON string (or empty).
#     Accepts python repr like "[{'content':...}]" and converts to JSON.
#     """
#     s = (s or "").strip()
#     if not s:
#         return ""

#     # already valid JSON?
#     try:
#         json.loads(s)
#         return s
#     except Exception:
#         pass

#     # try python literal -> json
#     try:
#         obj = ast.literal_eval(s)
#         return json.dumps(obj, ensure_ascii=False)
#     except Exception:
#         return ""

# def shorten_preview(s: str, max_chars: int = 300) -> str:
#     s = (s or "").replace("\r", "").strip()
#     if len(s) <= max_chars:
#         return s
#     return s[:max_chars] + " ..."

# def main():
#     if not IN_PATH.exists():
#         raise FileNotFoundError(f"Input file not found: {IN_PATH}")

#     repaired = 0
#     tuple_fixed = 0
#     bad_rows_merged = 0

#     with open(IN_PATH, "r", encoding="utf-8", newline="") as fin, open(
#         OUT_PATH, "w", encoding="utf-8", newline=""
#     ) as fout:
#         reader = csv.reader(fin, delimiter=",", quotechar='"', escapechar="\\")
#         writer = csv.writer(
#             fout,
#             delimiter=",",
#             quotechar='"',
#             quoting=csv.QUOTE_ALL,   # IMPORTANT: always quote everything => safe for newlines/commas
#             lineterminator="\n",
#         )

#         header = next(reader, None)
#         if header is None:
#             raise RuntimeError("Input CSV is empty")

#         # write expected header (force consistent schema)
#         writer.writerow(EXPECTED_HEADER)

#         for row in reader:
#             if not row:
#                 continue

#             if len(row) != 9:
#                 bad_rows_merged += 1

#             row = coerce_row_to_9_cols(row)

#             script, qid, qtype, question, answer, gold, nctx, ctx_prev, ctx_json = row

#             # Fix tuple-answers: answer field accidentally contains (answer, context)
#             if looks_like_tuple_answer(answer):
#                 ans_text, ctx_obj = try_extract_tuple_answer(answer)
#                 if ans_text is not None:
#                     answer = ans_text.strip()
#                     # If extracted context exists, write it into context_json
#                     ctx_json = json.dumps(ctx_obj, ensure_ascii=False)
#                     # Also make a preview from first context item if possible
#                     if isinstance(ctx_obj, list) and ctx_obj:
#                         first = ctx_obj[0]
#                         if isinstance(first, dict):
#                             ctx_prev = shorten_preview(first.get("content", ""))
#                     tuple_fixed += 1

#             # normalize context_json (python repr -> json, invalid -> empty)
#             ctx_json = normalize_context_json(ctx_json)

#             # fill n_context if missing but context_json exists
#             nctx = (nctx or "").strip()
#             if (not nctx or not nctx.isdigit()) and ctx_json:
#                 try:
#                     obj = json.loads(ctx_json)
#                     if isinstance(obj, list):
#                         nctx = str(len(obj))
#                 except Exception:
#                     pass
#             if not nctx:
#                 nctx = "0"

#             # if context_preview missing but context_json has data
#             if (not (ctx_prev or "").strip()) and ctx_json:
#                 try:
#                     obj = json.loads(ctx_json)
#                     if isinstance(obj, list) and obj and isinstance(obj[0], dict):
#                         ctx_prev = shorten_preview(obj[0].get("content", ""))
#                 except Exception:
#                     pass

#             writer.writerow([script, qid, qtype, question, answer, gold, nctx, ctx_prev, ctx_json])
#             repaired += 1

#     print(f"[OK] Input : {IN_PATH}")
#     print(f"[OK] Output: {OUT_PATH}")
#     print(f"[OK] Rows written: {repaired}")
#     print(f"[OK] Rows where columns were merged/padded: {bad_rows_merged}")
#     print(f"[OK] Tuple-answers fixed: {tuple_fixed}")

# if __name__ == "__main__":
#     main()
import csv
import ast
import json
from pathlib import Path

IN_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new_dataset.csv"
)
OUT_PATH = IN_PATH.with_name("answers_log_new_dataset_fixed.csv")

EXPECTED_HEADER = [
    "script",
    "question_id",
    "query_type",
    "question",
    "answer",
    "gold_answer",
    "n_context",
    "context_preview",
    "context_json",
]

PRINT_EVERY = 25  # progress print frequency


def looks_like_tuple_answer(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("(") and s.endswith(")")


def try_extract_tuple_answer(answer_field: str):
    try:
        obj = ast.literal_eval(answer_field)
        if isinstance(obj, tuple) and len(obj) == 2:
            ans, ctx = obj
            if isinstance(ans, str) and isinstance(ctx, (list, dict)):
                return ans, ctx
    except Exception:
        pass
    return None, None


def coerce_row_to_9_cols(row):
    if len(row) == 9:
        return row
    if len(row) > 9:
        head = row[:8]
        tail = row[8:]
        merged_last = ",".join(tail)
        return head + [merged_last]
    return row + [""] * (9 - len(row))


def normalize_context_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    try:
        json.loads(s)
        return s
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def shorten_preview(s: str, max_chars: int = 300) -> str:
    s = (s or "").replace("\r", "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " ..."


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v
    except ValueError:
        print("[WARN] Invalid integer. Using default.")
        return default


def ask_optional_int(prompt: str) -> int | None:
    raw = input(f"{prompt} [empty = no limit]: ").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v
    except ValueError:
        print("[WARN] Invalid integer. No limit will be applied.")
        return None


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    print("=== CSV Resume Repair Tool ===")
    print("Row numbers are DATA rows after the header (1 = first row after header).")
    start_row = ask_int("Start processing from data row", default=1)
    if start_row < 1:
        start_row = 1
    max_rows = ask_optional_int("Process at most N data rows (optional)")

    repaired = 0
    tuple_fixed = 0
    bad_rows_merged = 0

    skipped = 0
    processed = 0
    first_processed_rownum = None
    last_processed_rownum = None

    with open(IN_PATH, "r", encoding="utf-8", newline="") as fin, open(
        OUT_PATH, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin, delimiter=",", quotechar='"', escapechar="\\")
        writer = csv.writer(
            fout,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )

        header = next(reader, None)
        if header is None:
            raise RuntimeError("Input CSV is empty")

        writer.writerow(EXPECTED_HEADER)

        data_rownum = 0  # 1-based counter AFTER header

        for row in reader:
            data_rownum += 1

            if data_rownum < start_row:
                skipped += 1
                continue

            if max_rows is not None and processed >= max_rows:
                break

            if not row:
                continue

            if len(row) != 9:
                bad_rows_merged += 1

            row = coerce_row_to_9_cols(row)

            script, qid, qtype, question, answer, gold, nctx, ctx_prev, ctx_json = row

            # Fix tuple-answers: answer field accidentally contains (answer, context)
            if looks_like_tuple_answer(answer):
                ans_text, ctx_obj = try_extract_tuple_answer(answer)
                if ans_text is not None:
                    answer = ans_text.strip()
                    ctx_json = json.dumps(ctx_obj, ensure_ascii=False)
                    if isinstance(ctx_obj, list) and ctx_obj:
                        first = ctx_obj[0]
                        if isinstance(first, dict):
                            ctx_prev = shorten_preview(first.get("content", ""))
                    tuple_fixed += 1

            # normalize context_json
            ctx_json = normalize_context_json(ctx_json)

            # fill n_context if missing but context_json exists
            nctx = (nctx or "").strip()
            if (not nctx or not nctx.isdigit()) and ctx_json:
                try:
                    obj = json.loads(ctx_json)
                    if isinstance(obj, list):
                        nctx = str(len(obj))
                except Exception:
                    pass
            if not nctx:
                nctx = "0"

            # if context_preview missing but context_json has data
            if (not (ctx_prev or "").strip()) and ctx_json:
                try:
                    obj = json.loads(ctx_json)
                    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        ctx_prev = shorten_preview(obj[0].get("content", ""))
                except Exception:
                    pass

            writer.writerow([script, qid, qtype, question, answer, gold, nctx, ctx_prev, ctx_json])
            repaired += 1
            processed += 1

            if first_processed_rownum is None:
                first_processed_rownum = data_rownum
            last_processed_rownum = data_rownum

            if processed % PRINT_EVERY == 0:
                print(f"[PROGRESS] processed={processed} (data_row={data_rownum}) | skipped={skipped}")

    print("\n=== DONE ===")
    print(f"[OK] Input : {IN_PATH}")
    print(f"[OK] Output: {OUT_PATH}")
    print(f"[OK] Start row: {start_row}")
    print(f"[OK] Skipped data rows: {skipped}")
    print(f"[OK] First processed data row: {first_processed_rownum}")
    print(f"[OK] Last processed data row : {last_processed_rownum}")
    print(f"[OK] Rows written: {repaired}")
    print(f"[OK] Rows where columns were merged/padded: {bad_rows_merged}")
    print(f"[OK] Tuple-answers fixed: {tuple_fixed}")


if __name__ == "__main__":
    main()
