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

from pathlib import Path

import os
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

BASE_DIR_PATH = (
    PROJECT_ROOT
    / "main"
    / "out_aktuell"
)

BASE_DIR  =  Path(os.getenv("ANSWERS_LOG_PATH", str(BASE_DIR_PATH))).expanduser().resolve()
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
  
)

# Neo4j-GraphStore
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
database = "llmakg"   


graph_store = Neo4jPGStore(
    username=AUTH_USER ,
    password=AUTH_PASSWORD,
    url=URI,
    database=database,
)

#NUMBER_OF_ARTICLES = 250

index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

print("Knowledge Graph erfolgreich nach Neo4j exportiert!")
