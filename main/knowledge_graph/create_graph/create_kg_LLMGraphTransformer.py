import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"   # falls du eine eigene KG-DB hast

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

# OpenAI LLM (oder anderes Modell, das du nutzt)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# LLMGraphTransformer:
# - node_properties: welche Properties die erzeugten Knoten bekommen sollen
# - relationship_properties: Properties auf den Kanten
doc_transformer = LLMGraphTransformer(
    llm=llm,
    node_properties=["description"],
    relationship_properties=["description"],
)
BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell"
)

def load_chunks_from_jsonl(base_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for jsonl_path in base_dir.rglob("docling_chunks.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                text = row.get("text", "").strip()
                if not text:
                    continue

                metadata = {
                    "product": row.get("product"),
                    "product_category": row.get("product_category"),
                    "chunk_id": row.get("chunk_id"),
                    "source_file": str(jsonl_path),
                }

                docs.append(Document(page_content=text, metadata=metadata))
    return docs

def extract_graph_documents(docs: List[Document]):
    # Lässt das LLM Entity/Relation-Tripel aus den Texten erzeugen
    graph_docs = doc_transformer.convert_to_graph_documents(docs)
    print(f"Created {len(graph_docs)} graph documents.")
    return graph_docs

def write_graph_to_neo4j(graph_docs):
    """
    base_entity_label:
        • Standardmäßig heißt das Label der Entitäten "_Entity".
        • Du kannst z.B. "__Entity__" verwenden, wenn du konsistent mit Neo4j-GraphRAG bleiben willst.
    include_source:
        • Wenn True, wird für jeden Textchunk ein „Quell“-Knoten (z.B. :Document oder :Chunk)
          und eine Beziehung zur Entität angelegt.
    """
    graph.add_graph_documents(
        graph_docs
    )
    print("Graph documents written to Neo4j.")

def main():
    docs = load_chunks_from_jsonl(BASE_DIR)
    print(f"Loaded {len(docs)} text chunks.")

    graph_docs = extract_graph_documents(docs)
    write_graph_to_neo4j(graph_docs)

    print("Knowledge Graph construction finished.")

if __name__ == "__main__":
    main()

