import os
import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document

# -----------------------------
# 1) Define Properties 
# -----------------------------

# set OPENAI_API_KEY as environment variable 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())



URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "rag"        

BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell_advanced_rag"
)

INDEX_NAME = "rag_chunks"
NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"
EMB_PROPERTY = "embedding"

# -----------------------------
# 2) Chunks aus JSONL laden
# -----------------------------

def load_chunks_from_jsonl(base_dir: Path):
    docs = []
    for jsonl_path in base_dir.rglob("docling_chunks.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                text = row.get("text")
                product = row.get("product")
                product_category = row.get("product_category")
                file_name = row.get("file_name")

                if not text:
                    continue

                metadata = {
                    "product": product,
                    "product_category": product_category,
                    "file_name": str(file_name),
                }

                docs.append(Document(page_content=text, metadata=metadata))
    return docs



def main():
    print("Loading chunks from JSONL...")
    docs = load_chunks_from_jsonl(BASE_DIR)
    print(f"Loaded {len(docs)} chunks.")

    embeddings = OpenAIEmbeddings()  # z.B. text-embedding-3-large / ada-002 je nach Account

    print("Writing chunks + embeddings into Neo4j (this may take a while)...")

    vector_index = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=URI,
        username=AUTH_USER,
        password=AUTH_PASSWORD,
        database=DATABASE,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property=TEXT_PROPERTY,
        embedding_node_property=EMB_PROPERTY,
        # metadata wird automatisch als Properties gespeichert
    )

    print("Done. Vector index created in Neo4j:")
    print(f"  database = {DATABASE}")
    print(f"  index    = {INDEX_NAME}")
    print(f"  label    = {NODE_LABEL}")


if __name__ == "__main__":
    main()
