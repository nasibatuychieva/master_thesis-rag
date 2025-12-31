from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PGRetriever,
)

from main.evaluation.logger import log_antwort  
load_dotenv(find_dotenv())
# ---------------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------------

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)

username = os.getenv("NEO4J_USER")
password =os.getenv("NEO4J_PASSWORD")
uri = os.getenv("NEO4J_URI")
database=  "llmakg"
SCRIPT_NAME = "LlamaIndex_Vector_KG_Retriever"


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "golden_answers_dataset_short.jsonl"
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
# 4) Helper: Logging 
# ---------------------------------------------------------------------------

def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: Optional[List[Dict[str, Any]]] = None,
):
   
    log_antwort(
        script,
        question_id,
        query_type,
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )

# ---------------------------------------------------------------------------
# 5) Helpers: Extract content, metadata, id, score, type, source
# ---------------------------------------------------------------------------

def _extract_chunk_text(result_obj: Any) -> str:
    """
    Best-effort extraction of *Chunk node text* for logging + prompting.

    Priority:
      1) result_obj.node.text (TextNode / chunk)
      2) result_obj.node.get_content(metadata_mode="none")
      3) result_obj.get_content(metadata_mode="none") / get_content()
      4) str(result_obj)
    """
    # 1) NodeWithScore-like objects
    node = getattr(result_obj, "node", None)
    if node is not None:
        txt = getattr(node, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        
        try:
            node_content = node.get_content(metadata_mode="none")  
            if isinstance(node_content, str) and node_content.strip():
                return node_content.strip()
        except Exception:
            pass

        try:
            node_content = node.get_content() 
            if isinstance(node_content, str) and node_content.strip():
                return node_content.strip()
        except Exception:
            pass

    # 2) Direct get_content on result object
    try:
        c = result_obj.get_content(metadata_mode="none")  
        if isinstance(c, str) and c.strip():
            return c.strip()
    except Exception:
        pass

    try:
        c = result_obj.get_content()  
        if isinstance(c, str) and c.strip():
            return c.strip()
    except Exception:
        pass

    # 3) Last resort
    s = str(result_obj or "").strip()
    return s

def _extract_metadata_dict(result_obj: Any) -> Dict[str, Any]:
    """
    Try to find metadata dict on result or result.node.
    """
    meta = getattr(result_obj, "metadata", None)
    if isinstance(meta, dict):
        return meta

    node = getattr(result_obj, "node", None)
    meta2 = getattr(node, "metadata", None) if node is not None else None
    if isinstance(meta2, dict):
        return meta2

    return {}

def _extract_node_id(result_obj: Any) -> str:
    """
    Best-effort id extraction (result id or underlying node id).
    """
    for attr in ("id_", "id", "node_id", "ref_doc_id"):
        try:
            v = getattr(result_obj, attr, None)
            if v is not None and str(v).strip():
                return str(v).strip()
        except Exception:
            pass

    node = getattr(result_obj, "node", None)
    if node is not None:
        for attr in ("id_", "id", "node_id", "ref_doc_id"):
            try:
                v = getattr(node, attr, None)
                if v is not None and str(v).strip():
                    return str(v).strip()
            except Exception:
                pass

    return ""

def _extract_score(result_obj: Any) -> Any:
    try:
        return getattr(result_obj, "score", "")
    except Exception:
        return ""

def _infer_node_type(meta: Dict[str, Any], default: str = "chunk") -> str:
    """
    Provide something meaningful for logger's context_types_json.
    """

    for k in ("node_type", "label", "labels", "type", "__label__", "entity_type"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            return str(v[0])
    return default

def _infer_source(meta: Dict[str, Any]) -> str:
    return (
        meta.get("source")
        or meta.get("file_name")
        or meta.get("document")
        or meta.get("doc_id")
        or meta.get("ref_doc_id")
        or ""
    )

# ---------------------------------------------------------------------------
# 6) Helper: Answer with PG-Retriever (returns answer + context_items)
# ---------------------------------------------------------------------------

def answer_with_pg_retriever(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    results = pg_retriever.retrieve(question)

    context_items: List[Dict[str, Any]] = []
    context_lines: List[str] = []

    for r in results or []:
       
        content = _extract_chunk_text(r)
        content = (content or "").strip()
        if not content:
            continue

        meta = _extract_metadata_dict(r)
        rid = _extract_node_id(r)
        score = _extract_score(r)
        source = _infer_source(meta)
        node_type = _infer_node_type(meta, default="chunk")

        context_items.append(
            {
                "content": content,      # <- chunk text
                "source": source,
                "id": str(rid),
                "score": score,
                "node_type": node_type,  # <- enables context_types_json
            }
        )

    
        context_lines.append(f"- {content}")

   
    if not context_items:
        context_items = [
            {
                "content": "[NO CONTEXT RETURNED BY PG_RETRIEVER]",
                "source": "system",
                "id": "",
                "score": "",
                "node_type": "system",
            }
        ]
        context = context_items[0]["content"]
    else:
        context = "\n".join(context_lines)

    prompt = f"""
    "You are a technical support assistant for Arduino Products.\n"
    "Use ONLY the provided context. Do not use outside knowledge.\n"
    "If the context does not contain the answer, say exactly what information is missing.\n"
    "Answer in complete sentences.\n"
    "Answer as completely as possible.\n"
    "Adapt the structure and style of the answer to the type of the question "
    "(e.g., list items for 'which' questions, explain processes for 'how' questions, "
    "and compare variants for 'difference' questions).\n\n"

Context:
{context}

Question: {question}

Final answer:
"""
    answer = llm.complete(prompt).text.strip()
    return answer, context_items

# ---------------------------------------------------------------------------
# 7) Batch Mode (JSONL)
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
# 8) Manual Mode
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
# 9) Main Loop
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
