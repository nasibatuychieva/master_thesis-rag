import csv
import json
from pathlib import Path
import ast

# ---------------------------------------------------------
# Pfade anpassen
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation")

QUESTIONS_FILE = BASE_DIR / "questions.jsonl"
answers_log_new_FILE = BASE_DIR / "answers_log_new.csv"
JUDGE_RESULTS_FILE = BASE_DIR / "judge_results.csv"


# ---------------------------------------------------------
# 1. Fragen + Goldantworten laden
# ---------------------------------------------------------
def load_questions_map():
    """
    Lädt questions.jsonl und gibt ein Dict zurück:
    { question_id (str) : { "question": ..., "gold_answer": ..., "query_type": ... } }
    """
    qmap = {}
    
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            qid = str(item["id"])
            qmap[qid] = {
                "question": item["question"],
                "gold_answer": item.get("gold_answer", ""),
                "query_type": item.get("query_type", "")
            }

    return qmap


# ---------------------------------------------------------
# 2. LLM-Judge Prompt bauen
# ---------------------------------------------------------
def build_judge_prompt(question: str, gold_answer: str, candidate_answer: str) -> str:
    prompt = f"""
You are an expert evaluator for technical question answering in the Arduino domain. 
Your task is to compare model answers against a gold reference answer and judge their quality.

You MUST:
- Focus strictly on factual correctness and logical consistency with respect to the question and the gold answer.
- Ignore small differences in wording and style.
- Penalize contradictions, invented details, or statements that conflict with the gold answer or typical Arduino hardware documentation.
- Treat an answer as correct if it is semantically equivalent to the gold answer, even if phrased differently.

You also evaluate reasoning quality:
- "reasoning_correctness": Is the explanation technically and logically correct?
- "reasoning_depth": Does the answer use multi-step reasoning, relationships between entities (e.g., pins, timers, interfaces, sensors), or graph-like structure?
- "uses_relational_or_kg_evidence": Does the answer explicitly rely on relationships (e.g., A is part of B, A connects to B via C), as a knowledge graph or relational structure would?

Question:
{question}

Gold Answer:
{gold_answer}

Candidate Answer:
{candidate_answer}

Return a single JSON object with NO additional text, using exactly this schema:
{{
  "semantic_similarity": 0.0,
  "correctness": 0,
  "relevance": 0,
  "completeness": 0,
  "factual_accuracy": 0,
  "hallucination": "yes",
  "reasoning_correctness": 0,
  "reasoning_depth": 0,
  "uses_relational_or_kg_evidence": "unclear",
  "comments": "..."
}}
"""
    return prompt.strip()


# ---------------------------------------------------------
# 3. LLM-Aufruf (PLATZHALTER) – HIER musst du dein Modell anbinden
# ---------------------------------------------------------
import openai
import json
import ast

def call_llm_judge(prompt: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",  # oder dein Modell
        messages=[
            {"role": "system", "content": "You are a strict JSON-only evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # wie vorher: robust parsen
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(raw)
        except Exception:
            raise ValueError(f"Could not parse LLM output as JSON:\n{raw}")

    return result



# ---------------------------------------------------------
# 4. Antworten + Bewertungen verarbeiten
# ---------------------------------------------------------
def run_llm_judge():
    questions_map = load_questions_map()

    # answers_log_new.csv lesen
    with open(answers_log_new_FILE, "r", encoding="utf-8", newline="") as f_in, \
         open(JUDGE_RESULTS_FILE, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = [
    "script",
    "question_id",
    "query_type",
    "question",
    "gold_answer",
    "candidate_answer",
    "semantic_similarity",
    "correctness",
    "relevance",
    "completeness",
    "factual_accuracy",
    "hallucination",
    "reasoning_correctness",
    "reasoning_depth",
    "uses_relational_or_kg_evidence",
    "comments"
]


        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            script_name = row["script"]
            qid = row["question_id"].strip()
            candidate_question = row["question"]
            candidate_answer = row["answer"]

            if qid not in questions_map:
                print(f"Warnung: question_id {qid} nicht in questions.json gefunden – wird übersprungen.")
                continue

            gold = questions_map[qid]
            gold_question = gold["question"]
            gold_answer = gold["gold_answer"]
            query_type = gold["query_type"]

            # Prompt bauen
            prompt = build_judge_prompt(
                question=gold_question,
                gold_answer=gold_answer,
                candidate_answer=candidate_answer
            )

            # LLM aufrufen
            judge_result = call_llm_judge(prompt)

            # Zeile schreiben
            writer.writerow({
    "script": script_name,
    "question_id": qid,
    "query_type": query_type,
    "question": gold_question,
    "gold_answer": gold_answer,
    "candidate_answer": candidate_answer,
    "semantic_similarity": judge_result.get("semantic_similarity", ""),
    "correctness": judge_result.get("correctness", ""),
    "relevance": judge_result.get("relevance", ""),
    "completeness": judge_result.get("completeness", ""),
    "factual_accuracy": judge_result.get("factual_accuracy", ""),
    "hallucination": judge_result.get("hallucination", ""),
    "reasoning_correctness": judge_result.get("reasoning_correctness", ""),
    "reasoning_depth": judge_result.get("reasoning_depth", ""),
    "uses_relational_or_kg_evidence": judge_result.get("uses_relational_or_kg_evidence", ""),
    "comments": judge_result.get("comments", "")
    })


    print(f"Fertig. Ergebnisse gespeichert in: {JUDGE_RESULTS_FILE}")


# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
if __name__ == "__main__":
    run_llm_judge()
