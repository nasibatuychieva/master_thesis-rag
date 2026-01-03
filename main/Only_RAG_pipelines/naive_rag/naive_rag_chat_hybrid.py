import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document

from main.evaluation.logger import log_antwort

load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI_RAG")
AUTH_USER = os.getenv("NEO4J_USER_RAG")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD_RAG")
DATABASE = "rag"

INDEX_NAME = "rag_chunks"
NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"
EMB_PROPERTY = "embedding"

SCRIPT_NAME = "RAG_Naive_Hybrid"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_short.jsonl"
)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))
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

def retrieve_hybrid(question: str, k: int = 8) -> List[Document]:
    return vector_index.similarity_search(question, k=k)

def answer_naive_rag(
    question: str,
    *,
    k: int = 8,
) -> Tuple[str, List[Dict[str, Any]]]:

    docs = retrieve_hybrid(question, k=k)

    context = "\n\n".join(
        f"From file '{d.metadata.get('file_name','?')}':\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
You are an assistant answering questions about Arduino product documentation.

Use ONLY the following context.

Context:
{context}

Question:
{question}

Answer:
"""
    answer = llm.invoke(prompt).content.strip()

    context_items: List[Dict[str, Any]] = []
    for d in docs:
        context_items.append(
            {
                "content": d.page_content,
                "source": d.metadata.get("file_name", ""),
                "id": d.metadata.get("chunk_id", d.metadata.get("id", "")),
            }
        )

    return answer, context_items

def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: List[Dict[str, Any]],
):
    log_antwort(
        script,
        question_id,
        query_type,
        question,
        answer,
        gold_answer,
        context_items=context_items,
    )

def run_batch():
    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
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
            answer, context_items = answer_naive_rag(question)
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

def manual():
    qid = input("Question ID (optional): ").strip()
    question = input("Question: ").strip()
    gold = input("Gold answer (optional): ").strip()

    answer, context_items = answer_naive_rag(question)
    print("\nAnswer:\n", answer)

    safe_log(SCRIPT_NAME, qid, "manual", question, answer, gold, context_items)

def main():
    print("RAG_Naive_Hybrid")
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
