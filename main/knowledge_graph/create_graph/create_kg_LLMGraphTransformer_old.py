import os
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.documents.base import Document

# --- NEU: OpenAI statt lokales Llama ---
from langchain_openai import ChatOpenAI

# HuggingFace-Embeddings bleiben wie gehabt
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# 0) Einstellungen
# ----------------------------

BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell"
)

# ----------------------------
# OpenAI LLM (anstelle von ChatLlamaCpp)
# Voraussetzung: OPENAI_API_KEY ist in deiner .env gesetzt
# ----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # oder "gpt-4.1-mini", "gpt-4o", je nach Wunsch/Kosten
    temperature=0.0,
    # optional: max_tokens=2048
)

# Lokale Embeddings (all-MiniLM-L6-v2 -> 384 Dimensionen)
#USE_OPENAI_EMBEDDINGS = True  # via .env oder config

#if USE_OPENAI_EMBEDDINGS:
from langchain_openai import OpenAIEmbeddings
embedding_provider = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
EMBED_DIM = 1536
# else:
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     embedding_provider = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     EMBED_DIM = 384



URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "testmaster123"

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD 
) ## master2025
print("\n=== Connected to Neo4j Knowledge Graph ===\n")
# # LLMGraphTransformer mit OpenAI-LLM
doc_transformer = LLMGraphTransformer(llm=llm, node_properties=["description"], relationship_properties=["description"])

# ----------------------------
# 1) Constraints (einmalig)
# ----------------------------
graph.query("""
CREATE CONSTRAINT product_category_id IF NOT EXISTS
FOR (pc:ProductCategory) REQUIRE pc.id IS UNIQUE;
""")
graph.query("""
CREATE CONSTRAINT product_id IF NOT EXISTS
FOR (p:Product) REQUIRE p.id IS UNIQUE;
""")
graph.query("""
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
""")

# # ----------------------------
# # 2) JSONL-Iterator (zeilen-sicher)
# # ----------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# ----------------------------
# 3) Einzel-Chunk verarbeiten
# ----------------------------
import re

def normalize_product_name(name: str) -> str:
    if not name:
        return "UNKNOWN_PRODUCT"
    name = name.lower()
    name = name.replace("_", " ")
    name = re.sub(r"arduino®?", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()

def process_row(row: dict, fallback_idx: int):
    text = (row.get("text") or "").strip()
    if not text:
        return False
    product_raw = (row.get("product") or "UNKNOWN_PRODUCT").strip()

    product_category = (row.get("product_category") or "UNKNOWN_CATEGORY").strip()
    product          = normalize_product_name(product_raw)
    chunk_id         = (row.get("chunk_id") or f"chunk_{fallback_idx}").strip()

    file_name_raw    = (row.get("file_name") or "").strip()
    doc_id           = file_name_raw or "UNKNOWN_DOCUMENT"
    semantic_density = row.get("semantic_density")


    # 3a) Embedding
    embedding = embedding_provider.embed_query(text)

    # 3b) Upserts inkl. Document + sinnvolle 'name'-Properties
    graph.query(
        """
        MERGE (pc:ProductCategory {id: $pc_id})
          ON CREATE SET pc.name = $pc_id
        SET pc.name = coalesce(pc.name, $pc_id)

        MERGE (p:Product {id: $p_id})
          ON CREATE SET p.name = $p_id
        SET p.name = coalesce(p.name, $p_id)

        MERGE (c:Chunk {id: $chunk_id})
          ON CREATE SET c.text = $text
        SET  
             c.semantic_density  = $semantic_density,

             c.file_name         = $file_name,
             c.textEmbedding     = $embedding

        MERGE (p)-[:BELONGS_TO]->(pc)
        MERGE (p)<-[:MENTIONS]-(c)

        MERGE (d:Document {id: $doc_id})
          ON CREATE SET d.name = $doc_id
        SET d.name = coalesce(d.name, $doc_id)

        MERGE (c)-[:FROM_DOCUMENT]->(d)

        """,
        {
            "pc_id": product_category,
            "p_id": product,
            "chunk_id": chunk_id,
            "text": text,
            "semantic_density": semantic_density,
            "embedding": embedding,
            "file_name": file_name_raw,
            "doc_id": doc_id,
        }
    )

    # 3c) Entities & Relations extrahieren und Chunk verknüpfen
    lc_doc = Document(
        page_content=text,
        metadata={
            "product_category": product_category,
            "product": product,
            "chunk_id": chunk_id,   
            "file_name": file_name_raw,
            "doc_id": doc_id,
        }
    )

    graph_docs = doc_transformer.convert_to_graph_documents([lc_doc])

    if graph_docs:
        for gd in graph_docs:
            # 1) Entities speichern
            for n in gd.nodes:
    # Wir benutzen nur das Label :Entity
    # und legen n.type als Property ab
                graph.query(
            """
            MERGE (e:Entity {id: $ent_id})
            SET e += $props,
            e.entityType = $ent_type
            """,
            {
            "ent_id": n.id,
            "props": n.properties or {},
            "ent_type": n.type,
            }
        )


                # 2) Chunk -> HAS_ENTITY -> Entity
                graph.query(
                    """
                    MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $ent_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    {
                        "chunk_id": chunk_id,
                        "ent_id": n.id,
                    }
                )

            # 3) Relationen speichern
            for rel in gd.relationships:
                rel_type  = rel.type or "RELATED_TO"
                rel_props = rel.properties or {}   # description

                graph.query(
        f"""
        MATCH (s:Entity {{id: $src_id}}), (t:Entity {{id: $tgt_id}})
        MERGE (s)-[r:`{rel_type}`]->(t)
        SET  r += $props
        """,
        {
            "src_id": rel.source.id,
            "tgt_id": rel.target.id,
            "props": rel_props,
        }
    )


    return True

# ----------------------------
# 4) Alle .jsonl im Ordner verarbeiten
# ----------------------------
def process_dir(base_dir: Path):
    files = sorted(base_dir.rglob("*.jsonl"))
    if not files:
        print(f"Keine .jsonl-Dateien in {base_dir} gefunden.")
        return 0

    processed = 0
    for f in files:
        print(f"\n=== Datei: {f} ===")
        for i, row in enumerate(iter_jsonl(f), start=1):
            ok = process_row(row, fallback_idx=i)
            if ok:
                processed += 1
            if i % 50 == 0:
                print(f"  ... {i} Zeilen in {f.name} verarbeitet")
    return processed

total = process_dir(BASE_DIR)

# ----------------------------
# 5) Vector Index passend zur Embedding-Dimension
# ----------------------------
graph.query(f"""
CREATE VECTOR INDEX chunkVector IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {EMBED_DIM},
    `vector.similarity_function`: 'cosine'
  }}
}};
""")

print(f"\nDone. Processed {total} chunks aus {BASE_DIR}.\n")
