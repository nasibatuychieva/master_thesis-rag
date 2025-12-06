from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json

from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PGRetriever,
)

from main.evaluation.logger import log_antwort   # du nutzt bereits diese Funktion

# ---------------------------------------------------------------------------
# 1) Konfiguration
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0)

username = "neo4j"
password = "master2025"
uri = "neo4j://127.0.0.1:7687"
database = "llmakg"

SCRIPT_NAME = "KG_PGRetriever_Synonym_Vector"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset.jsonl"
)

# ---------------------------------------------------------------------------
# 2) GraphStore & Index
# ---------------------------------------------------------------------------

graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=uri,
    database=database,
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
)

# ---------------------------------------------------------------------------
# 3) Retriever
# ---------------------------------------------------------------------------

vector_retriever = VectorContextRetriever(
    graph_store=index.property_graph_store,
    embed_model=embed_model,
    similarity_top_k=10,
)

synonym_retriever = LLMSynonymRetriever(
    graph_store=index.property_graph_store,
    llm=llm,
)

pg_retriever = PGRetriever(
    sub_retrievers=[synonym_retriever, vector_retriever],
    llm=llm,
)

# ---------------------------------------------------------------------------
# 4) Helper: Logging (jetzt mit gold_answer)
# ---------------------------------------------------------------------------

def safe_log(script, question_id, query_type, question, answer, gold_answer):
    """
    Unified logging helper.
    """
    try:
        log_antwort(script, question_id, query_type, question, answer, gold_answer)
    except Exception:
        # absolute Fallback – zur Not ohne gold_answer
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            # minimaler Fallback – ohne IDs/Typ
            log_antwort(script, "", "", question, answer, "")



# ---------------------------------------------------------------------------
# 5) Helper: Answer with PG-Retriever
# ---------------------------------------------------------------------------

def answer_with_pg_retriever(question: str) -> str:
    results = pg_retriever.retrieve(question)

    context_lines = []
    for r in results:
        try:
            context_lines.append(f"- {r.get_content()}")
        except:
            context_lines.append(f"- {str(r)}")

    context = "\n".join(context_lines)

    prompt = f"""
You are an expert in Arduino hardware and embedded systems.
Answer the user question using ONLY the following retrieved KG context.

Context:
{context}

Question: {question}

Final answer:
"""
    return llm.complete(prompt).text.strip()

# ---------------------------------------------------------------------------
# 6) Batch Mode (JSONL)
# ---------------------------------------------------------------------------

def run_batch_from_file():
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset.jsonl not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] Invalid JSON at line {line_no}, skipped.")
                continue

            # Robust: id / question_id / query_id akzeptieren
            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id")
            question    = obj.get("question")
            gold_answer = obj.get("gold_answer")
            query_type  = obj.get("query_type")  # z.B. factual / relational / summary

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer = answer_with_pg_retriever(question)
            print(f"[ANSWER] {answer}\n")

            safe_log(SCRIPT_NAME, question_id, query_type, question, answer, gold_answer)

    print("\n[INFO] Batch processing completed.\n")


# ---------------------------------------------------------------------------
# 7) Manual Mode
# ---------------------------------------------------------------------------

def manual_question():
    qid = input("Question ID (optional): ").strip() or None
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or None

    answer = answer_with_pg_retriever(question)
    print("\nAnswer:\n", answer)

    safe_log(SCRIPT_NAME, qid, question, answer, gold_answer)

# ---------------------------------------------------------------------------
# 8) Main Loop
# ---------------------------------------------------------------------------

def main_loop():
    print("KG-PGRetriever (Synonym + Vector)")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question()
        elif mode in ("n", "no"):
            run_batch_from_file()
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main_loop()
