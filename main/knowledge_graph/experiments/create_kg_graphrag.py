from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import json

# LangChain Transformer
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_core.documents import Document

# LLM
from langchain_community.chat_models import ChatLlamaCpp

# LlamaIndex Neo4j Store
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# ----------------------------
# 1. Pfad zur JSONL-Datei
# ----------------------------
JSONL_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out\Mega\docling_chunks.jsonl"
)

# ----------------------------
# 2. LLM initialisieren
# ----------------------------
llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0
)

transformer = LLMGraphTransformer(llm=llm)

# ----------------------------
# 3. JSONL korrekt einlesen
# ----------------------------
docs = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text", "").strip()

        if text:
            docs.append(Document(page_content=text))

print(f"[INFO] Loaded {len(docs)} chunks.")

# ----------------------------
# 4. KG-Extraktion
# ----------------------------
graph_docs = transformer.convert_to_graph_documents(docs)
print(f"[INFO] Extracted triplets from {len(graph_docs)} chunks.")

# ----------------------------
# 5. Neo4j speichern
# ----------------------------
graph_store = Neo4jGraphStore(
    url="neo4j+s://0ac1e51a.databases.neo4j.io",
    username="neo4j",
    password="HsnGZbyrUW81nfnXYYFkTaYi8pYzS3VXE07SA4g1xZ",
)

graph_store.write_graph(graph_docs)

print("[INFO] Graph saved to Neo4j successfully.")
