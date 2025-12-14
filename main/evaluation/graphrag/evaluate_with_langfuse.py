"""
Offline-Evaluation mit LLM-as-a-Judge (5 Metriken)
- Liest QA-Paare aus answers_log.csv
- Ruft OpenAI (gpt-4o-mini) für 5 Metriken auf:
  Correctness, Relevance, Conciseness,
  Context Correctness, Hallucination
- Loggt alles in Langfuse (Trace + Scores)

CSV-Schema:
  script,question_id,query_type,question,answer,gold_answer
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

oa_client = OpenAI()          # nutzt OPENAI_API_KEY
langfuse = get_client()       # nutzt LANGFUSE_* Keys

CSV_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log.csv"
)

JUDGE_MODEL = "gpt-4o-mini"

# Score-Namen in Langfuse
SCORE_NAME_CORRECTNESS          = "qa_correctness_thesis"
SCORE_NAME_RELEVANCE            = "qa_relevance_thesis"
SCORE_NAME_CONCISENESS          = "qa_conciseness_thesis"
SCORE_NAME_CONTEXT_CORRECTNESS  = "qa_context_correctness_thesis"
SCORE_NAME_HALLUCINATION        = "qa_hallucination_thesis"


# -------------------------------------------------------------------
# 2) Gemeinsame Helper-Funktion: OpenAI aufrufen und 1–5 extrahieren
# -------------------------------------------------------------------

def call_openai_for_score(prompt: str) -> float:
    """
    Ruft OpenAI auf und erwartet eine Zahl 1–5 als Antwort.
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
        first_token = raw.split()[0]  # erstes Token (sollte eine Zahl sein)

        score_int = int(first_token)
        if score_int < 1 or score_int > 5:
            print(f"Warnung: Score außerhalb [1,5]: {score_int}, fallback = 3")
            return 3.0

        return float(score_int)

    except Exception as e:
        print(f"Fehler beim LLM-Aufruf: {e}. Fallback-Score = 3")
        return 3.0


# -------------------------------------------------------------------
# 3) Einzelne Metriken (jeweils anderer Prompt)
# -------------------------------------------------------------------

def judge_correctness(query: str, answer: str, gold: str) -> float:
    """
    Wie korrekt ist die Modellantwort im Vergleich zur Gold-Antwort?
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
    return call_openai_for_score(prompt)


def judge_relevance(query: str, answer: str, gold: str) -> float:
    """
    Wie relevant ist die Antwort für die Frage (inhaltliche Passung, nicht Stil)?
    """
    prompt = f"""
You are an evaluation model for question answering.

Query:
{query}

Model answer:
{answer}

Evaluate how relevant the model answer is to the given query.
Focus on whether the answer addresses the user's question.
Minor correctness issues are acceptable as long as the answer
is about the right topic.

Score from 1 to 5:
1 = not relevant at all
3 = partially relevant / mixed
5 = fully relevant and focused on the query

Return ONLY the number (1, 2, 3, 4, or 5).
"""
    return call_openai_for_score(prompt)


def judge_conciseness(query: str, answer: str, gold: str) -> float:
    """
    Wie prägnant ist die Antwort (nicht zu lang, nicht unnötig redundant)?
    """
    prompt = f"""
You are an evaluation model for answer conciseness.

Query:
{query}

Model answer:
{answer}

Evaluate how concise and to-the-point the model answer is,
while still covering the necessary information.

Score from 1 to 5:
1 = very verbose or rambling, lots of unnecessary text
3 = somewhat concise but contains some redundancy
5 = very concise and clear, no unnecessary information

Return ONLY the number (1, 2, 3, 4, or 5).
"""
    return call_openai_for_score(prompt)


def judge_context_correctness(query: str, answer: str, gold: str) -> float:
    """
    Wie gut bleibt die Antwort innerhalb der Fakten, die in der Gold-Antwort
    ausgedrückt werden? (Kein Widerspruch, keine erfundenen Details.)
    """
    prompt = f"""
You are an evaluation model for context faithfulness.

Query:
{query}

Model answer:
{answer}

Reference (gold answer):
{gold}

Evaluate how well the model answer stays faithful to the information
that could reasonably be derived from the reference. Penalize invented
facts that contradict the reference.

Score from 1 to 5:
1 = mostly unsupported or contradicting the reference
3 = partly consistent but with some unsupported details
5 = fully consistent and supported by the reference

Return ONLY the number (1, 2, 3, 4, or 5).
"""
    return call_openai_for_score(prompt)


def judge_hallucination(query: str, answer: str, gold: str) -> float:
    """
    Misst das Ausmaß von Halluzinationen (erfundene Fakten).
    5 = keine Halluzination, 1 = starke Halluzination.
    """
    prompt = f"""
You are an evaluation model for hallucination detection in QA.

Query:
{query}

Model answer:
{answer}

Reference (gold answer):
{gold}

Evaluate to what extent the model answer introduces information
that is not supported by the reference and is likely hallucinated.

Score from 1 to 5:
1 = heavily hallucinated, many unsupported or wrong details
3 = some unsupported or speculative statements
5 = no hallucination, all content is supported or safely inferred

Return ONLY the number (1, 2, 3, 4, or 5).
"""
    return call_openai_for_score(prompt)


# -------------------------------------------------------------------
# 4) Hauptfunktion: CSV einlesen, 5 Scores pro Zeile berechnen
# -------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        print("Raw fieldnames from CSV:", reader.fieldnames)

        for row in reader:
            # leichte Bereinigung
            clean_row = {}
            for k, v in row.items():
                if k is None:
                    continue
                key = k.strip().lstrip("\ufeff")
                value = (v or "").strip()
                clean_row[key] = value

            qid        = clean_row.get("question_id") or clean_row.get("id")
            question   = clean_row.get("question", "")
            answer     = clean_row.get("answer", "")
            gold       = clean_row.get("gold_answer", "")
            script     = clean_row.get("script", "")
            query_type = clean_row.get("query_type", "")

            if not question or not answer:
                print(f"Skip row {qid}: missing question or answer")
                continue

            with langfuse.start_as_current_observation(
                as_type="span",
                name="offline_eval_row",
            ) as span:

                # Input (für späteres Mapping im Export)
                span.update(
                    input={
                        "question_id": qid,
                        "script": script,
                        "query_type": query_type,
                        "question": question,
                        "answer": answer,
                        "gold_answer": gold,
                    }
                )

                # --- 5 Metriken berechnen (nacheinander, also 5 OpenAI-Calls) ---
                correctness   = judge_correctness(question, answer, gold)
                relevance     = judge_relevance(question, answer, gold)
                conciseness   = judge_conciseness(question, answer, gold)
                ctx_correct   = judge_context_correctness(question, answer, gold)
                hallucination = judge_hallucination(question, answer, gold)

                # Optional: alles im Output speichern
                span.update(
                    output={
                        "correctness": correctness,
                        "relevance": relevance,
                        "conciseness": conciseness,
                        "context_correctness": ctx_correct,
                        "hallucination": hallucination,
                    }
                )

                # --- Scores nach Langfuse schicken ---
                span.score_trace(
                    name=SCORE_NAME_CORRECTNESS,
                    value=correctness,
                    data_type="NUMERIC",
                )
                span.score_trace(
                    name=SCORE_NAME_RELEVANCE,
                    value=relevance,
                    data_type="NUMERIC",
                )
                span.score_trace(
                    name=SCORE_NAME_CONCISENESS,
                    value=conciseness,
                    data_type="NUMERIC",
                )
                span.score_trace(
                    name=SCORE_NAME_CONTEXT_CORRECTNESS,
                    value=ctx_correct,
                    data_type="NUMERIC",
                )
                span.score_trace(
                    name=SCORE_NAME_HALLUCINATION,
                    value=hallucination,
                    data_type="NUMERIC",
                )

                print(
                    f"Row {qid}: "
                    f"C={correctness}, R={relevance}, "
                    f"Con={conciseness}, Ctx={ctx_correct}, H={hallucination}"
                )

    print("Fertig – alle 5 Scores sind in Langfuse unter 'Scores' sichtbar.")


if __name__ == "__main__":
    main()
