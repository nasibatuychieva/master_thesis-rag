import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pathlib import Path
import os
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

# ------------------------------------------------------------------
# Neo4j-Connection
# ------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmagraphtrkg"

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

# ------------------------------------------------------------------
# Constraints 
# ------------------------------------------------------------------
graph.query("""
CREATE CONSTRAINT doc_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;
""")
graph.query("""
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
""")
graph.query("""
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;
""")

# ------------------------------------------------------------------
# LLM + Transformer
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    node_properties=["description"],
    relationship_properties=["description"],
)


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

BASE_DIR_PATH = (
    PROJECT_ROOT
    / "main"
    / "out_aktuell"
)

BASE_DIR  =  Path(os.getenv("ANSWERS_LOG_PATH", str(BASE_DIR_PATH))).expanduser().resolve()
# ------------------------------------------------------------------
# 1) Load Chunks 
# ------------------------------------------------------------------
def load_chunks_from_jsonl(base_dir: Path) -> List[Document]:
    docs: List[Document] = []


    for jsonl_path in base_dir.rglob("docling_chunks.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                text = (row.get("text") or "").strip()
                if not text:
                    continue

           
                file_name = (row.get("file_name") or "").strip()
                chunk_id = (row.get("chunk_id") or "").strip()

              
                if not file_name:
                    file_name = jsonl_path.name
                if not chunk_id:
                 
                    chunk_id = f"{file_name}_chunk"

                metadata = {
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "product": row.get("product"),
                    "product_category": row.get("product_category"),
                    "source_file": str(jsonl_path),
                }

                docs.append(Document(page_content=text, metadata=metadata))

    return docs

# ------------------------------------------------------------------
# 2) Create GraphDocuments from text chunks
# ------------------------------------------------------------------
def extract_graph_documents(docs: List[Document]):
    graph_docs = doc_transformer.convert_to_graph_documents(docs)
    print(f"Created {len(graph_docs)} graph documents.")
    return graph_docs

# ------------------------------------------------------------------
# 3) GraphDocuments to Neo4j
# ------------------------------------------------------------------
def write_graph_to_neo4j(graph_docs):
    for gd in graph_docs:
        src: Document = gd.source
        meta = src.metadata or {}
        text = src.page_content

        # Aus Metadata holen
        doc_id = (meta.get("file_name") or "UNKNOWN_DOCUMENT").strip()
        chunk_id = (meta.get("chunk_id") or "UNKNOWN_CHUNK").strip()


        graph.query(
            """
            MERGE (d:Document {id: $doc_id})
              ON CREATE SET d.name = $doc_id
            SET d.name = coalesce(d.name, $doc_id)
            """,
            {"doc_id": doc_id},
        )


        graph.query(
            """
            MERGE (c:Chunk {id: $chunk_id})
              ON CREATE SET c.text = $text
            SET c.file_name = $doc_id
            """,
            {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
            },
        )

   
        graph.query(
            """
            MATCH (c:Chunk {id: $chunk_id}), (d:Document {id: $doc_id})
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            {"chunk_id": chunk_id, "doc_id": doc_id},
        )

        for n in gd.nodes:
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
                },
            )

            # Chunk -> Entity
            graph.query(
                """
                MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $ent_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {
                    "chunk_id": chunk_id,
                    "ent_id": n.id,
                },
            )


        for rel in gd.relationships:
            rel_type = rel.type or "RELATED_TO"
            rel_props = rel.properties or {}

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
                },
            )

    print("Graph documents written to Neo4j with Document + Chunk nodes.")

# ------------------------------------------------------------------
# 4) main
# ------------------------------------------------------------------
def main():
    docs = load_chunks_from_jsonl(BASE_DIR)
    print(f"Loaded {len(docs)} text chunks.")

    graph_docs = extract_graph_documents(docs)
    write_graph_to_neo4j(graph_docs)

    print("Knowledge Graph construction finished.")

if __name__ == "__main__":
    main()
