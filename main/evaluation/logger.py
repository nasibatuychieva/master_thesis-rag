import csv
import os
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log.csv")

def log_antwort(script_name, question, answer, logfile=BASE_DIR):

    # Ensure logfile is a string path for os functions
    logfile = str(logfile)

    # Question ID extrahieren (alles vor dem ersten :)
    if ":" in question:
        question_id, question_text = question.split(":", 1)
        question_id = question_id.strip()
        question_text = question_text.strip()
    else:
        question_id = ""
        question_text = question

    file_exists = os.path.isfile(logfile)

    with open(logfile, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["script", "question_id", "question", "answer"])

        writer.writerow([script_name, question_id, question_text, answer])
