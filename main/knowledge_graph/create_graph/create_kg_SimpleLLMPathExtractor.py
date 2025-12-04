import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore

# ENV laden
load_dotenv(find_dotenv())

# Embedder & LLM
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0)

# JSONL-Dateien laden
BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell"
)

documents = []
for jsonl_path in BASE_DIR.rglob("*.jsonl"):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            txt = row.get("text")
            if not txt:
                continue
            metadata = {
                "file": jsonl_path.name,
                "product": row.get("product"),
                "product_category": row.get("product_category"),
                "chunk_id": row.get("chunk_id"),
            }
            documents.append(Document(text=txt, metadata=metadata))

print(f"{len(documents)} Dokumente geladen.")

# SimpleLLMPathExtractor
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=4,
    # show_progress=False,  # nur, falls deine Version diesen Parameter kennt
)

# Neo4j-GraphStore
username = "neo4j"
password = "master2025"
uri = "neo4j://127.0.0.1:7687"
database = "llmakg"

graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=uri,
    database=database,
)

NUMBER_OF_ARTICLES = 250

index = PropertyGraphIndex.from_documents(
    documents[:NUMBER_OF_ARTICLES],
    kg_extractors=[kg_extractor],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

print("Knowledge Graph erfolgreich nach Neo4j exportiert!")
