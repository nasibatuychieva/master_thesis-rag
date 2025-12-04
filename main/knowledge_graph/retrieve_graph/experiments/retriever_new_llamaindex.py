from dotenv import load_dotenv
load_dotenv()

import os
import re
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from langchain_community.graphs.neo4j_graph import Neo4jGraph  # nur noch für Cypher-Queries
from main.evaluation.logger import log_antwort
from llama_index.core import StorageContext, KnowledgeGraphIndex
# ---------- LlamaIndex / GraphRAG ----------
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Wenn du deinen graph_store woanders erzeugst, hier importieren:
# from main.graph_store import graph_store


# ---------------------------------------------------------------------------
# 1) LLM und Embeddings (LlamaIndex)
# ---------------------------------------------------------------------------

llm = LlamaCPP(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0.0,
    context_window=4096,
    max_new_tokens=1024,
    model_kwargs={
        "n_threads": 4,
        "n_gpu_layers": 0,
    },
)

embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# 3) WICHTIG: global setzen für LlamaIndex!
Settings.llm = llm
Settings.embed_model = embedding_model
# ---------------------------------------------------------------------------
# 2) Neo4j-Verbindung (für direkte Cypher-Queries, z.B. Kategorien)
# ---------------------------------------------------------------------------
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
)


graph_store = Neo4jGraphStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
)
print("[DEBUG] Knoten im Graph:",
      graph_store.query("MATCH (n) RETURN count(n) AS c"))


storage_context = StorageContext.from_defaults(
    graph_store=graph_store
)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    include_embeddings=True,
)

# Retriever daraus ableiten
retriever = kg_index.as_retriever(similarity_top_k=5)
query_engine = kg_index.as_query_engine()


# graph_rag_retriever = KnowledgeGraphRAGRetriever(
#     storage_context=storage_context, 
#     embed_model=embedding_model,
#     llm=llm,
#     verbose=True,
# )

# # Neuer Stil: kein ResponseSynthesizer mehr, QueryEngine übernimmt das intern
# query_engine = RetrieverQueryEngine.from_args(
#     retriever=graph_rag_retriever
# )




def answer_question(question: str) -> str:
    """
    Dynamische Entscheidung:
      - Wenn Frage nach 'Welche Produkte gehören zu Kategorie/Familie X?' aussieht
        → direkte Graph-Abfrage (Cypher, global, ALLE Produkte).
      - Sonst → KnowledgeGraphRAGRetriever (LlamaIndex GraphRAG).
    """
    print("[DEBUG] Frage:", question)


    response = query_engine.query(question)
    print("[DEBUG] response-Objekt:", type(response), response)

    answer_text = str(response)
    print("[DEBUG] answer_text:", repr(answer_text))



# ---------------------------------------------------------------------------
# 6) CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("RAG + Knowledge Graph Chat. Enter 'exit' for finishing.\n")
    SCRIPT_NAME = "retriever_new_llamaindex.py"  # oder os.path.basename(__file__)

    while (q := input("> ")).strip().lower() != "exit":
        if not q:
            continue

        out = answer_question(q)
        print("\n" + out + "\n")

        # Antwort + Frage loggen
        log_antwort(SCRIPT_NAME, q, out)
