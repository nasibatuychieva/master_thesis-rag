# ============================================================================
# ONLY_RAG_Advanced
# Retrieval-Augmented Generation (Neo4j Vector + LLM Rerank)
# with FULL logging for LLM-as-a-Judge (faithfulness-ready)
# ============================================================================

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document

from main.evaluation.logger import log_antwort

# ----------------------------------------------------------------------------
# 0) Environment
# ----------------------------------------------------------------------------

load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "rag"

INDEX_NAME = "rag_chunks"
NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"
EMB_PROPERTY = "embedding"

SCRIPT_NAME = "RAG_Vector_Advanced"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

# ----------------------------------------------------------------------------
# 1) Models
# ----------------------------------------------------------------------------

llm_router = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_rerank = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_index = Neo4jVector(
    embedding=embeddings,
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    database=DATABASE,
    index_name=INDEX_NAME,
    node_label=NODE_LABEL,
    text_node_property=TEXT_PROPERTY,
    embedding_node_property=EMB_PROPERTY,
)

# ----------------------------------------------------------------------------
# 2) Pre-Retrieval
# ----------------------------------------------------------------------------

def pre_retrieval(query: str) -> Dict[str, Any]:
    rewrite_prompt = f"""
Rewrite the following technical question to be concise, precise,
and unambiguous while preserving its meaning.

Question:
{query}
"""
    rewritten = llm_router.invoke(rewrite_prompt).content.strip()

    expand_prompt = f"""
Given the following technical query:

{rewritten}

List 3–8 important keywords or short phrases (comma-separated, no explanations).
"""
    expansion = llm_router.invoke(expand_prompt).content
    keywords = [k.strip() for k in expansion.split(",") if k.strip()]

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "keywords": keywords,
    }

# ----------------------------------------------------------------------------
# 3) Retrieval
# ----------------------------------------------------------------------------

def retrieve_chunks(pre: Dict[str, Any], k: int = 12) -> List[Document]:
    retrieval_query = (
        pre["rewritten_query"]
        + "\n\nRelevant keywords: "
        + ", ".join(pre["keywords"])
    )
    return vector_index.similarity_search(retrieval_query, k=k)

# ----------------------------------------------------------------------------
# 4) Rerank + Context Fusion
# ----------------------------------------------------------------------------

def rerank_and_select(
    pre: Dict[str, Any], docs: List[Document], top_k: int = 6
) -> List[Document]:
    if not docs:
        return []

    doc_blocks = []
    for i, d in enumerate(docs):
        doc_blocks.append(
            f"[DOC {i} | file={d.metadata.get('file_name','?')}]\n{d.page_content}"
        )

    rerank_prompt = f"""
You are a reranker for technical documentation QA.

User query:
{pre["rewritten_query"]}

Documents:
{chr(10).join(doc_blocks)}

Task:
Return ONLY a comma-separated list of the {top_k} most relevant document indices.
"""
    ranked = llm_rerank.invoke(rerank_prompt).content

    indices = []
    for tok in ranked.replace("\n", ",").split(","):
        if tok.strip().isdigit():
            idx = int(tok.strip())
            if 0 <= idx < len(docs):
                indices.append(idx)

    if not indices:
        indices = list(range(min(top_k, len(docs))))

    return [docs[i] for i in indices[:top_k]]

# ----------------------------------------------------------------------------
# 5) Final Answer + CONTEXT ITEMS
# ----------------------------------------------------------------------------

def answer_advanced_rag(
    question: str,
    *,
    retrieve_k: int = 12,
    top_k: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:

    pre = pre_retrieval(question)
    retrieved_docs = retrieve_chunks(pre, k=retrieve_k)
    selected_docs = rerank_and_select(pre, retrieved_docs, top_k=top_k)

    # ---- Build fused context used for answering ----
    fused_context = "\n\n".join(
        f"From file '{d.metadata.get('file_name','?')}', "
        f"product '{d.metadata.get('product','?')}':\n{d.page_content}"
        for d in selected_docs
    )

    answer_prompt = f"""
You are an assistant answering questions about Arduino product documentation.

Question:
{pre["original_query"]}

Context:
{fused_context}

Rules:
- Answer concisely (max 5 sentences).
- Use ONLY information supported by the context.
- If something is missing, say:
  "The documentation snippet does not contain this information."
"""
    answer = llm_answer.invoke(answer_prompt).content.strip()

    # ---- CONTEXT ITEMS FOR FAITHFULNESS ----
    context_items: List[Dict[str, Any]] = []

    for d in selected_docs:
        context_items.append({
            "content": d.page_content,
            "source": d.metadata.get("file_name", ""),
            "product": d.metadata.get("product", ""),
            "id": d.metadata.get("chunk_id", d.metadata.get("id", "")),
        })

    # optional: fused context (debug)
    context_items.append({
        "content": fused_context,
        "source": "fused_context",
        "id": "",
    })

    return answer, context_items

# ----------------------------------------------------------------------------
# 6) Logging Helper
# ----------------------------------------------------------------------------

def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: List[Dict[str, Any]],
):
    try:
        log_antwort(
            script,
            question_id,
            query_type,
            question,
            answer,
            gold_answer,
            context_items=context_items,
        )
    except Exception as e:
        print("[WARN] logging failed:", e)
        log_antwort(script, "", "", question, answer, "")

# ----------------------------------------------------------------------------
# 7) Batch Mode
# ----------------------------------------------------------------------------

def run_batch():
    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            question = obj.get("question")
            if not question:
                continue

            question_id = str(obj.get("id", ""))
            query_type = obj.get("query_type", "")
            gold_answer = obj.get("gold_answer", "")

            print(f"[QID {question_id}] {question}")
            answer, context_items = answer_advanced_rag(question)
            print(answer, "\n")

            safe_log(
                SCRIPT_NAME,
                question_id,
                query_type,
                question,
                answer,
                gold_answer,
                context_items,
            )

# ----------------------------------------------------------------------------
# 8) Manual Mode
# ----------------------------------------------------------------------------

def manual():
    qid = input("Question ID (optional): ").strip()
    question = input("Question: ").strip()
    gold = input("Gold answer (optional): ").strip()

    answer, context_items = answer_advanced_rag(question)
    print("\nAnswer:\n", answer)

    safe_log(SCRIPT_NAME, qid, "manual", question, answer, gold, context_items)

# ----------------------------------------------------------------------------
# 9) Main
# ----------------------------------------------------------------------------

def main():
    print("ONLY_RAG_Advanced")
    print("y = manual | n = batch | exit")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("exit", "quit", "q"):
            break
        elif cmd in ("y", "yes"):
            manual()
        elif cmd in ("n", "no"):
            run_batch()

if __name__ == "__main__":
    main()
