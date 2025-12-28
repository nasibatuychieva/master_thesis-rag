from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple

from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PGRetriever,
)

from main.evaluation.logger import log_antwort  # expects context_items kwarg in your new logger

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

SCRIPT_NAME = "LlamaIndex_Vector_KG_Retriever"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
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
# 4) Helper: Logging (mit context_items)
# ---------------------------------------------------------------------------

def safe_log(
    script: str,
    question_id: Optional[str],
    query_type: Optional[str],
    question: str,
    answer: str,
    gold_answer: Optional[str],
    context_items: Optional[List[Dict[str, Any]]] = None,
):
    """
    Unified logging helper with backward-compatible fallback if logger signature differs.
    """
    try:
        log_antwort(
            script,
            question_id,
            query_type,
            question,
            answer,
            gold_answer or "",
            context_items=context_items,
        )
    except TypeError:
        # older logger without context_items kwarg
        log_antwort(script, question_id, query_type, question, answer, gold_answer or "")
    except Exception:
        # absolute fallback
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            log_antwort(script, "", "", question, answer, "")

# ---------------------------------------------------------------------------
# 5) Helper: Answer with PG-Retriever (returns answer + context_items)
# ---------------------------------------------------------------------------

def answer_with_pg_retriever(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    results = pg_retriever.retrieve(question)

    context_items: List[Dict[str, Any]] = []
    context_lines: List[str] = []

    for r in results or []:
        # Best-effort content extraction
        try:
            content = r.get_content()
        except Exception:
            content = str(r)

        content = (content or "").strip()
        if not content:
            continue

        # Best-effort id / score / source extraction
        rid = ""
        score: Any = ""
        source = ""

        try:
            rid = getattr(r, "id_", "") or getattr(r, "id", "") or ""
        except Exception:
            rid = ""

        try:
            score = getattr(r, "score", "")
        except Exception:
            score = ""

        # Some LlamaIndex nodes store metadata on r or r.node
        meta = None
        try:
            meta = getattr(r, "metadata", None)
        except Exception:
            meta = None

        if meta is None:
            try:
                node = getattr(r, "node", None)
                meta = getattr(node, "metadata", None) if node is not None else None
            except Exception:
                meta = None

        if isinstance(meta, dict):
            source = (
                meta.get("source")
                or meta.get("file_name")
                or meta.get("document")
                or meta.get("doc_id")
                or ""
            )

        context_items.append({
            "content": content,
            "source": source,
            "id": str(rid),
            "score": score,
        })

        context_lines.append(f"- {content}")

    # If nothing retrieved, log explicit marker for judge/debug
    if not context_items:
        context_items = [{
            "content": "[NO CONTEXT RETURNED BY PG_RETRIEVER]",
            "source": "system",
            "id": "",
            "score": "",
        }]
        context = context_items[0]["content"]
    else:
        context = "\n".join(context_lines)

    prompt = f"""
You are an expert in Arduino hardware and embedded systems.
Answer the user question using ONLY the following retrieved KG context.

Context:
{context}

Question: {question}

Final answer:
"""
    answer = llm.complete(prompt).text.strip()
    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch Mode (JSONL)
# ---------------------------------------------------------------------------

def run_batch_from_file():
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset_new.jsonl not found.")
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

            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id") or ""
            question    = obj.get("question") or ""
            gold_answer = obj.get("gold_answer") or ""
            query_type  = obj.get("query_type") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_with_pg_retriever(question)
            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] n_context={len([c for c in context_items if c.get('content')])}\n")

            safe_log(
                SCRIPT_NAME,
                str(question_id),
                str(query_type),
                question,
                answer,
                gold_answer,
                context_items=context_items,
            )

    print("\n[INFO] Batch processing completed.\n")

# ---------------------------------------------------------------------------
# 7) Manual Mode
# ---------------------------------------------------------------------------

def manual_question():
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional, e.g., factual/relational/summary): ").strip() or "manual"
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or ""

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_pg_retriever(question)
    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] n_context={len([c for c in context_items if c.get('content')])}\n")

    safe_log(
        SCRIPT_NAME,
        qid,
        qtype,
        question,
        answer,
        gold_answer,
        context_items=context_items,
    )

# ---------------------------------------------------------------------------
# 8) Main Loop
# ---------------------------------------------------------------------------

def main_loop():
    print("KG-PGRetriever (Synonym + Vector) + Context Logging")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

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
