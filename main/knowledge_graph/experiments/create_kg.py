import os
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.documents.base import Document
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI



# ----------------------------
# 0) Einstellungen
# ----------------------------
# Ordner, der mehrere .jsonl-Dateien enthält (rekursiv durchsuchen)

BASE_DIR = Path(r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell")

# LLM (lokales Qwen-GGUF via ChatLlamaCpp)
llm = ChatOpenAI(
    model="gpt-4o",   # oder gpt-4o-mini / gpt-4.1-mini
    temperature=0,
)

embedding_provider = OpenAIEmbeddings(model="text-embedding-ada-002")
EMBED_DIM = 384  # wichtig: zum Vector Index passend

graph = Neo4jGraph(
    url="bolt://localhost:7687",    # oder neo4j+s://<id>.databases.neo4j.io für Aura
    username="neo4j",
    password="testmaster123"
)

doc_transformer = LLMGraphTransformer(llm=llm)

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

# ----------------------------
# 2) JSONL-Iterator (zeilen-sicher)
# ----------------------------
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
    name = name.lower()                          # alles klein
    name = name.replace("_", " ")                # Unterstriche → Leerzeichen
    name = re.sub(r"arduino®?", "", name)        # optional Branding entfernen
    name = re.sub(r"[^a-z0-9\s]", "", name)      # Sonderzeichen entfernen
    name = re.sub(r"\s+", " ", name).strip()     # Leerzeichen glätten
    return name.title()  

def process_row(row: dict, fallback_idx: int):
    text = (row.get("text") or "").strip()
    if not text:
        return False
    product_raw = (row.get("product") or "UNKNOWN_PRODUCT").strip()

    product_category = (row.get("product_category") or "UNKNOWN_CATEGORY").strip()
    product          = normalize_product_name(product_raw)
    chunk_id         = (row.get("chunk_id") or f"chunk_{fallback_idx}").strip()

    # NEU: file_name direkt aus JSONL nehmen
    file_name_raw    = (row.get("file_name") or "").strip()
    # Wenn du die Endung entfernen willst, nimm Path(...).stem, ansonsten so lassen:
    doc_id           = file_name_raw or "UNKNOWN_DOCUMENT"

    section          = row.get("section")
    semantic_density = row.get("semantic_density")
    element          = row.get("element")
    tutorial         = row.get("tutorial")

    # 3a) Embedding
    embedding = embedding_provider.embed_query(text)

    # 3b) Upserts inkl. Document + sinnvolle 'name'-Properties
    graph.query("""
    MERGE (pc:ProductCategory {id: $pc_id})
      ON CREATE SET pc.name = $pc_id
    SET pc.name = coalesce(pc.name, $pc_id)

    MERGE (p:Product {id: $p_id})
      ON CREATE SET p.name = $p_id
    SET p.name = coalesce(p.name, $p_id)

    MERGE (c:Chunk {id: $chunk_id})
      ON CREATE SET c.text = $text
    SET  c.section           = $section,
         c.semantic_density  = $semantic_density,
         c.element           = $element,
         c.tutorial          = $tutorial,
         c.file_name         = $file_name,
         c.textEmbedding     = $embedding

    MERGE (pc)-[:HAS_PRODUCT]->(p)
    MERGE (p)<-[:PART_OF]-(c)

    // Document-Knoten
    MERGE (d:Document {id: $doc_id})
      ON CREATE SET d.name = $doc_id
    SET d.name = coalesce(d.name, $doc_id)

    MERGE (c)-[:IN_DOCUMENT]->(d)

    // Tutorial nur, wenn tutorial wirklich gesetzt ist
    WITH p, c, $tutorial AS tut
    WHERE tut IS NOT NULL AND trim(tut) <> ''

    MERGE (t:Tutorial {id: tut})
      ON CREATE SET t.name = tut
    SET t.name = coalesce(t.name, tut)

    MERGE (p)-[:HAS_TUTORIAL]->(t)
    MERGE (c)-[:IN_TUTORIAL]->(t)
    """,
    {
        "pc_id": product_category,
        "p_id": product,            # dein normalisierter Produktname
        "chunk_id": chunk_id,
        "text": text,
        "section": section,
        "semantic_density": semantic_density,
        "element": element,
        "tutorial": tutorial,      # direkt aus JSONL
        "embedding": embedding,
        "file_name": file_name_raw,
        "doc_id": doc_id,
    }
)


    # 3c) Entities & Relations extrahieren und Chunk verknüpfen (unverändert)
    lc_doc = Document(
        page_content=text,
        metadata={
            "product_category": product_category,
            "product": product,
            "chunk_id": chunk_id,
            "section": section,
            "element": element,
            "tutorial": tutorial,
            "file_name": file_name_raw,
            "doc_id": doc_id,
        }
    )
    graph_docs = doc_transformer.convert_to_graph_documents([lc_doc])

    if graph_docs:
        for gd in graph_docs:
            chunk_node = Node(id=chunk_id, type="Chunk")
            for node in gd.nodes:
                gd.relationships.append(
                    Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                )
        graph.add_graph_documents(graph_docs)

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
