"""
Offline-Evaluation mit LLM-as-a-Judge (Variante A)
- Liest QA-Paare aus answers_log_new.csv
- Ruft OpenAI (gpt-4o-mini) auf, um einen Score 1–5 zu berechnen
- Loggt alles in Langfuse (Trace + Score)

Voraussetzungen:
- OPENAI_API_KEY in .env
- LANGFUSE_SECRET_KEY und LANGFUSE_PUBLIC_KEY in .env
- answers_log_new.csv mit Spalten: script,question_id,question,answer,gold_answer
"""

from pathlib import Path
import csv
from typing import Optional

from dotenv import load_dotenv, find_dotenv

from openai import OpenAI
from langfuse import get_client

# -------------------------------------------------------------------
# 1) Environment & Clients
# -------------------------------------------------------------------

load_dotenv(find_dotenv())

# OpenAI-Client (nutzt OPENAI_API_KEY aus .env)
oa_client = OpenAI()

# Langfuse-Client (v3 SDK)
langfuse = get_client()

# Pfad zu deiner CSV-Datei
CSV_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new.csv"
)

# Name des Scores, wie er in Langfuse angezeigt werden soll
SCORE_NAME = "qa_correctness_thesis"

# Welches OpenAI-Modell für den Judge verwendet wird
JUDGE_MODEL = "gpt-4o-mini"


# -------------------------------------------------------------------
# 2) LLM-Judge Funktion
# -------------------------------------------------------------------

def judge_with_llm(query: str, answer: str, gold: str) -> float:
    """
    Ruft OpenAI auf, um einen Score zwischen 1 und 5 zu bestimmen.
    Bewertet: "Wie korrekt ist die Modellantwort im Vergleich zur Gold-Antwort?"
    """

    prompt = f"""
You are an evaluation model for question answering.

Query:
{query}

Model answer:
{answer}

Gold answer:
{gold}

Evaluate how correct the model answer is compared to the gold answer.
Ignore minor wording differences if the meaning is the same.

Score from 1 to 5:
1 = completely incorrect
3 = partially correct
5 = fully correct

Return ONLY the number (1, 2, 3, 4, or 5).
"""

    try:
        resp = oa_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        raw_content: Optional[str] = resp.choices[0].message.content
        if raw_content is None:
            print("Warnung: LLM hat keine Antwort geliefert, fallback = 3")
            return 3.0

        raw = raw_content.strip()
        # Nur die erste Zeile / das erste Token nehmen
        first_token = raw.split()[0]

        score_int = int(first_token)
        # Sicherheitscheck: nur Werte 1–5 zulassen
        if score_int < 1 or score_int > 5:
            print(f"Warnung: Score außerhalb [1,5]: {score_int}, fallback = 3")
            return 3.0

        return float(score_int)

    except Exception as e:
        # Bei Rate-Limit o.Ä. nicht alles abbrechen, sondern fallback
        print(f"Fehler beim LLM-Aufruf: {e}. Fallback-Score = 3")
        return 3.0


# -------------------------------------------------------------------
# 3) Hauptfunktion: CSV durchgehen und bewerten
# -------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Spaltennamen an deine CSV anpassen
            qid = row.get("question_id") or row.get("id")
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            gold = row.get("gold_answer", "").strip()

            if not question or not answer:
                print(f"Skip row {qid}: missing question or answer")
                continue

            # 1) Observation / Span in Langfuse starten
            with langfuse.start_as_current_observation(
                as_type="span",
                name="offline_eval_row",
            ) as span:

                # 2) Input (Frage, Antwort, Gold) im Span speichern
                span.update(
                    input={
                        "question_id": qid,
                        "question": question,
                        "answer": answer,
                        "gold_answer": gold,
                    }
                )

                # 3) LLM-Judge ausführen
                score_value = judge_with_llm(question, answer, gold)

                # Optional: Output im Span speichern (z.B. für Debug)
                span.update(
                    output={
                        "qa_correctness_score": score_value,
                    }
                )

                # 4) Score als numerischen Score in Langfuse speichern
                span.score_trace(
                    name=SCORE_NAME,
                    value=score_value,
                    data_type="NUMERIC",
                )

                print(f"Row {qid}: Score = {score_value}")

    print(
        "Fertig – Scores sind in Langfuse unter 'Scores' sichtbar "
        f"(Score Name = {SCORE_NAME})."
    )


# -------------------------------------------------------------------
# 4) Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
