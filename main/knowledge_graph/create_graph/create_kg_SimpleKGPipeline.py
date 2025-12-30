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
DATABASE = "simplekg"

from pathlib import Path

import os
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

BASE_DIR_PATH = (
    PROJECT_ROOT
    / "main"
    / "out_aktuell"
)

BASE_DIR  =  Path(os.getenv("ANSWERS_LOG_PATH", str(BASE_DIR_PATH))).expanduser().resolve()

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM für Schema + KG-Build
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
    model_params={"temperature": 0}
)


embedder = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


schema_extractor = SchemaFromTextExtractor(llm=llm)


# ---------------------------------------------------------------------------
# 2) Helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    """Zeilenweise JSONL lesen und jeweils als dict zurückgeben."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# 3) KG-Build
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

   
    document_metadata = {
        "file_name": file_name or "UNKNOWN_FILE",
        "chunk_id": chunk_id,
    
        "product": (row.get("product") or "").strip() or None,
        "product_category": (row.get("product_category") or "").strip() or None,
    }


    await kg_builder.run_async(
        text=text,
        document_metadata=document_metadata,
    
    )

    return True


# ---------------------------------------------------------------------------
# 4) Alle JSONL-Dateien im BASE_DIR verarbeiten
# ---------------------------------------------------------------------------

async def process_dir_async(base_dir: Path):
 


    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,

        from_pdf=False,
        neo4j_database=DATABASE,
        on_error="RAISE",          #
    )


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
