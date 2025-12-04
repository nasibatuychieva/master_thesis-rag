import os
import json
import re
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
import asyncio
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
import os
import asyncio
from typing import Dict

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------



# -----------------------------
# 1) Settings
# -----------------------------
load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "rag"

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

schema_extractor = SchemaFromTextExtractor(
    llm=OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0}
    )
)

# -----------------------------
# 2) Chat Loop
# -----------------------------

async def chat_loop_async():
    print("Advanced RAG chat. Type 'exit' to quit.")
    while True:
        q = input("\nFrage: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        extracted_schema = await schema_extractor.run(text=q)
        print("\nExtracted schema:\n", extracted_schema)

if __name__ == "__main__":
    asyncio.run(chat_loop_async())








# Extract the schema from the text

# ---------------------------------------------------------------------------
# 3) Schema Definition
# ---------------------------------------------------------------------------
# print("connected to neo4j")
# NODE_TYPES = [
#     {
#     "label": "Product",
#     "properties": [
#         {"name": "name", "type": "STRING"}
#     ]
# },
# {
#     "label": "ProductCategory",
#     "properties": [
#         {"name": "name", "type": "STRING"}
#     ]
# },
#     {
#         "label": "Board",
#         "properties": [
#             {"name": "name", "type": "STRING"},
#             {"name": "family", "type": "STRING"},
#         ],
#     },
#     {
#         "label": "Component",  # Prozessor, IMU, Sensor, Connector, Power-IC
#         "properties": [
#             {"name": "name", "type": "STRING"},
#             {"name": "category", "type": "STRING"},
#         ],
#     },
#     {
#         "label": "Interface",  # Digital Pin, Analog Pin, I2C, SPI, Bluetooth etc.
#         "properties": [
#             {"name": "name", "type": "STRING"},
#             {"name": "direction", "type": "STRING"},  # INPUT / OUTPUT / INOUT
#         ],
#     },
#     {
#         "label": "Feature",
#         "properties": [
#             {"name": "name", "type": "STRING"},
#         ],
#     },
#     {
#         "label": "TargetArea",  # z.B. "Wearables", "IoT", "Robotics"
#         "properties": [
#             {"name": "name", "type": "STRING"},
#         ],
#     },
#     {
#         "label": "Spec",  # Recommended Operating Conditions, Power, Dimensions etc.
#         "properties": [
#             {"name": "name", "type": "STRING"},
#             {"name": "value", "type": "STRING"},
#             {"name": "unit", "type": "STRING"},
#         ],
#     },
#     {
#         "label": "Accessory",  # Shields, Zusatzboards
#         "properties": [
#             {"name": "name", "type": "STRING"},
#         ],
#     },
# ]

# RELATIONSHIP_TYPES = [
#     "HAS_COMPONENT",  
#     "BELONGS_TO",      # (Board)-[:HAS_COMPONENT]->(Component)
#     "HAS_INTERFACE",      # (Board)-[:HAS_INTERFACE]->(Interface)
#     "HAS_FEATURE",        # (Board)-[:HAS_FEATURE]->(Feature)
#     "TARGETED_AT",        # (Board)-[:TARGETED_AT]->(TargetArea)
#     "HAS_SPEC",           # (Board)-[:HAS_SPEC]->(Spec)
#     "COMPATIBLE_WITH",    # (Board)-[:COMPATIBLE_WITH]->(Accessory)
#     "CONNECTS_TO",        # (Interface)-[:CONNECTS_TO]->(Interface)
#     "USES_COMPONENT",     # (Board)-[:USES_COMPONENT]->(Component)
#     "POWERED_VIA",        # (Board)-[:POWERED_VIA]->(Spec)  (z.B. Input Voltage)
#     "SUPPLIES_POWER",     # (Component)-[:SUPPLIES_POWER]->(Interface)
#     "CONTROLS",           # (Board)-[:CONTROLS]->(Component)
#     "SUPPORTS",           # (Board)-[:SUPPORTS]->(Interface) (z.B. Bluetooth)
# ]

# # Patterns: wie dürfen Knoten via Relationship verbunden sein?
# PATTERNS = [
#     ("Board", "HAS_COMPONENT", "Component"),
#     ("Board", "HAS_INTERFACE", "Interface"),
#     ("Board", "HAS_FEATURE", "Feature"),
#     ("Board", "TARGETED_AT", "TargetArea"),
#     ("Board", "HAS_SPEC", "Spec"),
#     ("Board", "COMPATIBLE_WITH", "Accessory"),
#     ("Interface", "CONNECTS_TO", "Interface"),
#     ("Board", "USES_COMPONENT", "Component"),
#     ("Board", "POWERED_VIA", "Spec"),
#     ("Component", "SUPPLIES_POWER", "Interface"),
#     ("Board", "CONTROLS", "Component"),
#     ("Board", "SUPPORTS", "Interface"),
# ]

# POTENTIAL_SCHEMA = {
#     "node_types": NODE_TYPES,
#     "relationship_types": RELATIONSHIP_TYPES,
#     "patterns": PATTERNS,
#     # keine weiteren Node-Typen erfinden:
#     "additional_node_types": False,
# }

# # ---------------------------------------------------------------------------
# # 4) SimpleKGPipeline erstellen
# # ---------------------------------------------------------------------------

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    from_pdf=False,          # wir geben Text direkt, keine PDFs
    potential_schema=POTENTIAL_SCHEMA,
    entities=NODE_TYPES,
    relationships=RELATIONSHIP_TYPES,
      # oder "RAISE" für Debugging
)

# # ---------------------------------------------------------------------------
# # 5) Hilfsfunktionen: JSONL lesen, Produktnamen normalisieren
# # ---------------------------------------------------------------------------


# def iter_jsonl(path: Path):
#     """Zeilenweise JSONL lesen und jeweils als dict zurückgeben."""
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)

# def normalize_product_name(name: str) -> str:
#     if not name:
#         return "UNKNOWN_PRODUCT"
#     name = name.lower()
#     name = name.replace("_", " ")
#     name = re.sub(r"arduino®?", "", name)
#     name = re.sub(r"[^a-z0-9\s]", "", name)
#     name = re.sub(r"\s+", " ", name).strip()
#     return name.title()



