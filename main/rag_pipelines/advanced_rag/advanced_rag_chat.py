import os
from typing import List, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from main.evaluation.logger import log_antwort  
# -----------------------------
# 1) Settings
# -----------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "rag"  

INDEX_NAME = "rag_chunks"
NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"
EMB_PROPERTY = "embedding"

# LLMs für verschiedene Schritte
llm_router = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_rerank = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

embeddings = OpenAIEmbeddings()

# VectorStore aus bestehendem Index
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


# -----------------------------
# 2) Pre-Retrieval
# -----------------------------

def pre_retrieval(query: str) -> Dict:
    """Query Rewriting + simple Query Expansion."""
    # 1) Rewrite
    rewrite_prompt = f"""
You are a query rewriting assistant.
Rewrite the following user question so that it is concise, unambiguous,
and technically precise, while keeping its meaning unchanged.

User question:
{query}
"""
    rewritten = llm_router.invoke(rewrite_prompt).content.strip()

    # 2) Expansion: wichtige Begriffe
    expand_prompt = f"""
Given the following technical query:

{rewritten}

List 3 to 8 important keywords or short phrases (no explanations),
comma-separated, that are useful for retrieving relevant documentation.
"""
    expansion = llm_router.invoke(expand_prompt).content
    keywords = [k.strip() for k in expansion.split(",") if k.strip()]

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "keywords": keywords,
    }


# -----------------------------
# 3) Retrieval
# -----------------------------

def retrieve_chunks(pre: Dict, k: int = 12) -> List[Document]:
    """Vector-Retrieval aus Neo4j mit Rewrite + Keywords."""
    rq = pre["rewritten_query"]
    kw = ", ".join(pre["keywords"])

    retrieval_query = f"{rq}\n\nRelevant keywords: {kw}"
    docs = vector_index.similarity_search(retrieval_query, k=k)
    return docs


# -----------------------------
# 4) Post-Retrieval (Rerank + Fusion)
# -----------------------------

def post_retrieval(pre: Dict, docs: List[Document], top_k: int = 6) -> str:
    """Rerank + Fusion/Summary -> ein gemeinsamer Kontext-String."""
    if not docs:
        return ""

    # a) Reranking mit LLM
    doc_texts = []
    for i, d in enumerate(docs):
        product = d.metadata.get("product", "Unknown product")
        file_name = d.metadata.get("file_name", "unknown_file")
        doc_texts.append(f"[DOC {i} | product: {product} | file: {file_name}]\n{d.page_content}")

    joined_docs = "\n\n".join(doc_texts)

    rerank_prompt = f"""
You are a reranker for retrieval-augmented generation.

User query:
{pre["rewritten_query"]}

Below are retrieved document chunks from technical product documentation,
each labelled [DOC i].

Documents:
{joined_docs}

Task:
1. Identify the {top_k} most relevant documents for answering the user query.
2. Output ONLY a comma-separated list of their indices i (e.g., "0, 3, 5").

Do not output anything else.
"""
    ranked = llm_rerank.invoke(rerank_prompt).content
    indices = []
    for tok in ranked.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(docs):
                indices.append(idx)

    if not indices:
        indices = list(range(min(top_k, len(docs))))

    selected_docs = [docs[i] for i in indices[:top_k]]

    # b) Fusion/Summary: ein gemeinsamer Kontext
    fusion_text = "\n\n".join(
        f"From file '{d.metadata.get('file_name', 'unknown_file')}', "
        f"product '{d.metadata.get('product', 'Unknown')}':\n"
        f"{d.page_content}"
        for d in selected_docs
)


    fusion_prompt = f"""
You are a helpful assistant that prepares context for a Retrieval-Augmented
Generation (RAG) system.

User query:
{pre["original_query"]}

Below are the most relevant document chunks from a product documentation corpus:

{fusion_text}

Task:
Produce a single coherent, non-redundant summary that preserves all technical
details necessary to answer the user query later. Do not answer the query yet;
only create a high-quality context summary.
"""
    fused_context = llm_rerank.invoke(fusion_prompt).content.strip()
    return fused_context


# -----------------------------
# 5) Finale Antwort
# -----------------------------

def answer_advanced_rag(query: str) -> str:
    pre = pre_retrieval(query)
    docs = retrieve_chunks(pre)
    fused_context = post_retrieval(pre, docs)

    answer_prompt = f"""
You are an assistant answering questions about Arduino product documentation.

User question:
{pre["original_query"]}

Context (summarized from retrieved documentation):
{fused_context}

Answer the user's question in as much technical detail as needed.
If something is not covered by the context, explicitly say so.
"""
    answer = llm_answer.invoke(answer_prompt).content.strip()
    source_files = []
    for d in docs:
        file_name = d.metadata.get("file_name")
        if file_name:
            source_files.append(file_name)

    unique_sources = sorted(set(source_files))

    sources_block = "\n".join(f"- {s}" for s in unique_sources)

    return answer + "\n\nSources:\n" + sources_block



# -----------------------------
# 6) Einfacher CLI-Loop
# -----------------------------

import os
from typing import List, Dict
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from main.evaluation.logger import log_antwort  
import json
# -----------------------------
# 1) Settings
# -----------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "rag"  

INDEX_NAME = "rag_chunks"
NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"
EMB_PROPERTY = "embedding"

# LLMs für verschiedene Schritte
llm_router = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_rerank = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

embeddings = OpenAIEmbeddings()

# VectorStore aus bestehendem Index
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


# -----------------------------
# 2) Pre-Retrieval
# -----------------------------

def pre_retrieval(query: str) -> Dict:
    """Query Rewriting + simple Query Expansion."""
    # 1) Rewrite
    rewrite_prompt = f"""
You are a query rewriting assistant.
Rewrite the following user question so that it is concise, unambiguous,
and technically precise, while keeping its meaning unchanged.

User question:
{query}
"""
    rewritten = llm_router.invoke(rewrite_prompt).content.strip()

    # 2) Expansion: wichtige Begriffe
    expand_prompt = f"""
Given the following technical query:

{rewritten}

List 3 to 8 important keywords or short phrases (no explanations),
comma-separated, that are useful for retrieving relevant documentation.
"""
    expansion = llm_router.invoke(expand_prompt).content
    keywords = [k.strip() for k in expansion.split(",") if k.strip()]

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "keywords": keywords,
    }


# -----------------------------
# 3) Retrieval
# -----------------------------

def retrieve_chunks(pre: Dict, k: int = 12) -> List[Document]:
    """Vector-Retrieval aus Neo4j mit Rewrite + Keywords."""
    rq = pre["rewritten_query"]
    kw = ", ".join(pre["keywords"])

    retrieval_query = f"{rq}\n\nRelevant keywords: {kw}"
    docs = vector_index.similarity_search(retrieval_query, k=k)
    return docs


# -----------------------------
# 4) Post-Retrieval (Rerank + Fusion)
# -----------------------------

def post_retrieval(pre: Dict, docs: List[Document], top_k: int = 6) -> str:
    """Rerank + Fusion/Summary -> ein gemeinsamer Kontext-String."""
    if not docs:
        return ""

    # a) Reranking mit LLM
    doc_texts = []
    for i, d in enumerate(docs):
        product = d.metadata.get("product", "Unknown product")
        file_name = d.metadata.get("file_name", "unknown_file")
        doc_texts.append(f"[DOC {i} | product: {product} | file: {file_name}]\n{d.page_content}")

    joined_docs = "\n\n".join(doc_texts)

    rerank_prompt = f"""
You are a reranker for retrieval-augmented generation.

User query:
{pre["rewritten_query"]}

Below are retrieved document chunks from technical product documentation,
each labelled [DOC i].

Documents:
{joined_docs}

Task:
1. Identify the {top_k} most relevant documents for answering the user query.
2. Output ONLY a comma-separated list of their indices i (e.g., "0, 3, 5").

Do not output anything else.
"""
    ranked = llm_rerank.invoke(rerank_prompt).content
    indices = []
    for tok in ranked.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(docs):
                indices.append(idx)

    if not indices:
        indices = list(range(min(top_k, len(docs))))

    selected_docs = [docs[i] for i in indices[:top_k]]

    # b) Fusion/Summary: ein gemeinsamer Kontext
    fusion_text = "\n\n".join(
        f"From file '{d.metadata.get('file_name', 'unknown_file')}', "
        f"product '{d.metadata.get('product', 'Unknown')}':\n"
        f"{d.page_content}"
        for d in selected_docs
)


    fusion_prompt = f"""
You are a helpful assistant that prepares context for a Retrieval-Augmented
Generation (RAG) system.

User query:
{pre["original_query"]}

Below are the most relevant document chunks from a product documentation corpus:

{fusion_text}

Task:
Produce a single coherent, non-redundant summary that preserves all technical
details necessary to answer the user query later. Do not answer the query yet;
only create a high-quality context summary.
"""
    fused_context = llm_rerank.invoke(fusion_prompt).content.strip()
    return fused_context


# -----------------------------
# 5) Finale Antwort
# -----------------------------

def answer_advanced_rag(query: str) -> str:
    pre = pre_retrieval(query)
    docs = retrieve_chunks(pre)
    fused_context = post_retrieval(pre, docs)

    answer_prompt = f"""
You are an assistant answering questions about Arduino product documentation.

User question:
{pre["original_query"]}

Context (summarized from retrieved documentation):
{fused_context}

Answer the user's question in as much technical detail as needed.
If something is not covered by the context, explicitly say so.
"""
    answer = llm_answer.invoke(answer_prompt).content.strip()
    source_files = []
    for d in docs:
        file_name = d.metadata.get("file_name")
        if file_name:
            source_files.append(file_name)

    unique_sources = sorted(set(source_files))

    sources_block = "\n".join(f"- {s}" for s in unique_sources)

    return answer + "\n\nSources:\n" + sources_block



# -----------------------------
# 6) Einfacher CLI-Loop
# -----------------------------

def safe_log(script, question_id, query_type, question, answer, gold_answer):
    """
    Unified logging helper.
    """
    try:
        log_antwort(script, question_id, query_type, question, answer, gold_answer)
    except Exception:
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            log_antwort(script, "", "", question, answer, "")

SCRIPT_NAME = "ONLY_RAG_Advanced"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset.jsonl"
)

def answer_with_rag(question: str, top_k: int = 20) -> str:
    """
    Verwendet deine neue RAG-Pipeline (rag.search), um eine Antwort zu generieren.
    """
    response = answer_advanced_rag(question)
    return response


def run_batch_from_file(top_k: int = 20):
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

            # id / question_id / query_id robust behandeln
            question_id = obj.get("id") 
            query_type  = obj.get("query_type") 
            question    = obj.get("question")
            gold_answer = obj.get("gold_answer")
            # factual / relational / summary / ...

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer = answer_with_rag(question, top_k=top_k)
            print(f"[ANSWER] {answer}\n")

            # Einheitliches Logging
            safe_log(SCRIPT_NAME, question_id, query_type, question, answer, gold_answer)

    print("\n[INFO] Batch processing completed.\n")

def manual_question(top_k: int = 20):
    qid = input("Question ID (optional): ").strip() or None
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or None

    if not question:
        print("Empty question, skipping.\n")
        return

    answer = answer_with_rag(question, top_k=top_k)
    print("\nAnswer:\n", answer, "\n")

    safe_log(SCRIPT_NAME, qid, question, answer, gold_answer)

def main_loop(top_k: int = 20):
    print("Retrieve_kg_SimpleKGPipeline (VectorCypher + GraphRAG)")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question(top_k=top_k)
        elif mode in ("n", "no"):
            run_batch_from_file(top_k=top_k)
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

if __name__ == "__main__":

    main_loop(top_k=5)
 
