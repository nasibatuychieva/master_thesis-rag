import os
from pathlib import Path
from typing import Dict
from main.evaluation.logger import log_antwort
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"

# Neo4j-Driver (DB wird über ENV oder default gewählt)
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM für Schema + KG-Build
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"temperature": 0}
)

# Embedder für KG-Embeddings (für Knoten/Kanten)
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate


retrieval_query = """
WITH node, score

// lokale Produkte an diesem Chunk (MENTIONS in beliebiger Richtung)
OPTIONAL MATCH (node)-[:MENTIONS]-(p0:Product)

// Kategorien dieser Produkte (Property "category" auf Product)
WITH node, score,
     collect(DISTINCT p0) AS local_products,
     collect(DISTINCT p0.category) AS cats

UNWIND cats AS cat
OPTIONAL MATCH (p_all:Product {category: cat})

// weitere Entities zum Kontext
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(e:__Entity__)

// alles wieder einsammeln (und local_products im Scope behalten!)
WITH node, score,
     local_products,
     collect(DISTINCT cat)    AS categories,
     collect(DISTINCT p_all) AS products_in_category,
     collect(DISTINCT e)     AS entities

RETURN DISTINCT
  node.text AS text,
  score     AS score,
  categories,
  [p IN products_in_category | p.name] AS products_in_category,
  [p IN local_products        | p.name] AS local_products,
  [en IN entities             | en.name] AS entities


"""


# Create retriever
retriever = VectorCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
rag = GraphRAG(retriever=retriever, llm=llm)

# Search

def chat_loop(top_k: int = 20):
    print("Retrieve_kg_SimpleKGPipeline. Type your question or exit for quitting.\n")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        q_lower = question.lower()

        # Standard: GraphRAG über Chunks
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": top_k},
            return_context=True,
        )
        answer = response.answer  
        print("\nAntwort:\n")
        print(answer)

 
        # Logging für RAG Antwort
        log_antwort("LLMGraphTransformer_Retriever", question, answer)



if __name__ == "__main__":
    try:
        chat_loop(top_k=5)
    finally:
        driver.close()
