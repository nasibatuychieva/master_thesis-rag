from __future__ import annotations

import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.prompts import PromptTemplate


def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def main() -> None:
    load_dotenv(find_dotenv())

    uri = _need("NEO4J_URI")
    user = _need("NEO4J_USER")
    pwd = _need("NEO4J_PASSWORD")
    _need("OPENAI_API_KEY")

    database = os.getenv("NEO4J_DATABASE", "llmakg")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    _ = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model=model, temperature=0)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()

    graph_store = Neo4jGraphStore(username=user, password=pwd, url=uri, database=database)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    rag_retriever = KnowledgeGraphRAGRetriever(storage_context=storage_context, verbose=True)

    ANSWER_PROMPT = PromptTemplate(
        "You are a technical support assistant for Arduino Products.\n"
        "Use ONLY the provided context. Do not use outside knowledge.\n"
        "If the context does not contain the answer, say exactly what information is missing.\n"
        "Answer in complete sentences.\n"
        "Answer as completely as possible.\n"
        "Context:\n{context_str}\n\n"
        "Question:\n{query_str}\n\n"
        "Answer:\n"
    )

    qe = RetrieverQueryEngine.from_args(
        retriever=rag_retriever,
        llm=llm,
        text_qa_template=ANSWER_PROMPT,
        response_mode="compact",
    )

    r = qe.query("What is Arduino Uno WiFi Rev2?")
    print(str(getattr(r, "response", r)).strip())

    driver.close()


if __name__ == "__main__":
    main()
