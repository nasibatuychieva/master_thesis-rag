import os
import json
import asyncio
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "neo4j"

BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\out_aktuell"
)

# Neo4j-Driver (DB wird über ENV oder default gewählt)
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM für Schema + KG-Build
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
    model_params={"temperature": 0}
)

# Embedder für KG-Embeddings (für Knoten/Kanten)
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Komponente zum automatischen Schema-Extrahieren
schema_extractor = SchemaFromTextExtractor(llm=llm)


# ---------------------------------------------------------------------------
# 2) Hilfsfunktionen: JSONL lesen
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    """Zeilenweise JSONL lesen und jeweils als dict zurückgeben."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


async def build_schema_from_corpus(base_dir: Path) -> Dict:
    """
    Liest einige Beispieltexte aus dem Korpus und lässt daraus
    automatisch ein Schema extrahieren.
    """
    sample_texts = []
    for jsonl_path in base_dir.rglob("*.jsonl"):
        for row in iter_jsonl(jsonl_path):
            text = (row.get("text") or "").strip()
            if text:
                sample_texts.append(text)
            if len(sample_texts) >= 200:
                break
        if len(sample_texts) >= 200:
            break

    if not sample_texts:
        raise ValueError("Keine Texte gefunden, um ein Schema zu extrahieren.")

    schema_input = "\n\n".join(sample_texts)
    print("Extrahiere Schema aus Beispieldaten ...")
    extracted_schema = await schema_extractor.run(text=schema_input)
    print("Schema extrahiert.")
    return extracted_schema


# ---------------------------------------------------------------------------
# 3) KG-Build für eine einzelne Zeile
# ---------------------------------------------------------------------------

async def process_row(row: dict, kg_builder: SimpleKGPipeline, fallback_idx: int) -> bool:
    """
    Verarbeitet eine einzelne JSONL-Zeile:
    - nimmt 'text' als Inhalt
    - verwendet 'file_name' (+ optional 'chunk_id') als Dokument-/Chunk-Metadaten
    """
    text = (row.get("text") or "").strip()
    if not text:
        return False

    file_name = (row.get("file_name") or "").strip()
    chunk_id = (row.get("chunk_id") or f"chunk_{fallback_idx}").strip()

    # Metadaten für den Dokument-/Chunk-Knoten in Neo4j
    document_metadata = {
        "file_name": file_name or "UNKNOWN_FILE",
        "chunk_id": chunk_id,
        # optional weitere Felder, falls vorhanden:
        "product": (row.get("product") or "").strip() or None,
        "product_category": (row.get("product_category") or "").strip() or None,
    }

    # Aufruf der Pipeline:
    # - text  -> Entitäten + Relationen extrahieren
    # - document_metadata -> :Document / Chunk-Knoten mit diesen Properties
    await kg_builder.run_async(
        text=text,
        document_metadata=document_metadata,
        # wichtig, wenn nicht Default-DB
    )

    return True


# ---------------------------------------------------------------------------
# 4) Alle JSONL-Dateien im BASE_DIR verarbeiten
# ---------------------------------------------------------------------------

async def process_dir_async(base_dir: Path):
    # 1) Schema einmalig aus dem Korpus extrahieren
    extracted_schema = await build_schema_from_corpus(base_dir)
    

    # 2) Pipeline mit diesem Schema aufsetzen
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=extracted_schema,   # automatisch extrahiertes Schema
        from_pdf=False,
        neo4j_database=DATABASE,
        on_error="RAISE",          # für Debug; später ggf. "IGNORE"
    )

    # 3) Alle *.jsonl-Dateien durchgehen
    files = sorted(base_dir.rglob("*.jsonl"))
    if not files:
        print(f"Keine .jsonl-Dateien in {base_dir} gefunden.")
        return 0

    processed = 0
    for f in files:
        print(f"\n=== Datei: {f} ===")
        for i, row in enumerate(iter_jsonl(f), start=1):
            try:
                ok = await process_row(row, kg_builder, fallback_idx=i)
            except Exception as e:
                print(f"  Fehler in Zeile {i} von {f.name}: {e}")
                ok = False

            if ok:
                processed += 1

            if i % 50 == 0:
                print(f"  ... {i} Zeilen in {f.name} verarbeitet")

    print(f"\nFERTIG – insgesamt {processed} Chunks verarbeitet.")
    return processed


# ---------------------------------------------------------------------------
# 5) main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(process_dir_async(BASE_DIR))
    finally:
        driver.close()
