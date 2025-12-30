from __future__ import annotations

import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from llama_index.embeddings.openai import OpenAIEmbedding

# -------------------------
# Config
# -------------------------
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "simplekg")  

COMMUNITY_LABEL = "__Community__"
COMM_LEVEL_PROP = "level"
COMM_SUMMARY_PROP = "summary"
COMM_FULL_PROP = "full_content"
COMM_SUMMARY_KEYWORD ="keywords"
COMM_EMB_PROP = "embedding"

BATCH = 200
FULL_CONTENT_CHAR_LIMIT = 8000

embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def build_comm_text(summary: str | None, full: str | None) -> str:
    summary = (summary or "").strip()
    full = (full or "").strip()
    if len(full) > FULL_CONTENT_CHAR_LIMIT:
        full = full[:FULL_CONTENT_CHAR_LIMIT]
    if summary and full:
        return summary + "\n\n" + full
    return summary or full


def embed_level1_communities(driver) -> None:
    fetch_cypher = f"""
    MATCH (c:{COMMUNITY_LABEL})
    WHERE c.{COMM_LEVEL_PROP} = 1
      AND c.{COMM_EMB_PROP} IS NULL
      AND (c.{COMM_SUMMARY_PROP} IS NOT NULL OR c.{COMM_FULL_PROP} IS NOT NULL OR c.{COMM_SUMMARY_KEYWORD} IS NOT NULL)
    RETURN elementId(c) AS cid, c.{COMM_SUMMARY_PROP} AS summary, c.{COMM_FULL_PROP} AS full, c.{COMM_SUMMARY_KEYWORD} AS keywords
    LIMIT $limit
    """

    update_cypher = f"""
    UNWIND $rows AS row
    MATCH (c:{COMMUNITY_LABEL})
    WHERE elementId(c) = row.cid
    SET c.{COMM_EMB_PROP} = row.embedding
    """

    total = 0
    while True:
        with driver.session(database=DATABASE) as session:
            rows = session.run(fetch_cypher, limit=BATCH).data()

        if not rows:
            print(f"[DONE] Embedded {total} level-1 communities.")
            return

        payload = []
        for r in rows:
            text = build_comm_text(r.get("summary"), r.get("full"))
            if not text.strip():
                continue
            vec = embed_model.get_text_embedding(text)
            payload.append({"cid": r["cid"], "embedding": vec})

        with driver.session(database=DATABASE) as session:
            session.run(update_cypher, rows=payload)

        total += len(payload)
        print(f"[OK] Embedded {total} level-1 communities so far...")


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()
    try:
        embed_level1_communities(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()

# CREATE VECTOR INDEX communityEmbedding_simplekg IF NOT EXISTS
# FOR (c:__Community__)
# ON (c.embedding)
# OPTIONS {
#   indexConfig: {
#     `vector.dimensions`: 1536,
#     `vector.similarity_function`: 'cosine'
#   }
# };
