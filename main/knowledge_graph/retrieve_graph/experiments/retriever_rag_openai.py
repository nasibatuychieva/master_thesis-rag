import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG


# ----------------------------
# Neo4j Connection
# ----------------------------
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "testmaster123"

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))

# ----------------------------
# Embeddings + Retriever + LLM
# ----------------------------
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

retriever = VectorRetriever(
    driver,
    index_name="chunkVector",
    embedder=embedder,
)

llm = OpenAILLM(model_name="gpt-4o")

# GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)


# ----------------------------
# CLI LOOP (Input from Terminal)
# ----------------------------
print("GraphRAG Chat. Type 'exit' to quit.\n")

while True:
    question = input("> ").strip()

    if question.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not question:
        continue

    # RAG Search
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": 5}
    )

    print("\nAnswer:\n")
    print(response.answer)
    print("\n" + "-"*80 + "\n")


driver.close()
