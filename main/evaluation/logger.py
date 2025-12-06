import csv
import os
from pathlib import Path

BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new.csv"
)

def log_antwort(
    script_name,
    question_id,
    query_type,
    question,
    answer,
    gold_answer="",
    logfile=BASE_DIR,
):
    """
    Logs:
        script, question_id, query_type, question, answer, gold_answer
    """

    logfile = str(logfile)

    # falls question_id/query_type/gold_answer None sind → leere Spalten
    if question_id is None:
        question_id = ""
    if query_type is None:
        query_type = ""
    if gold_answer is None:
        gold_answer = ""

    file_exists = os.path.isfile(logfile)

    with open(logfile, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        # Header nur einmal schreiben
        if not file_exists:
            writer.writerow(
                ["script", "question_id", "query_type", "question", "answer", "gold_answer"]
            )

        writer.writerow(
            [script_name, question_id, query_type, question, answer, gold_answer]
        )
