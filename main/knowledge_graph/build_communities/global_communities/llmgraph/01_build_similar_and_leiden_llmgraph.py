import os
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER =  "neo4j"
AUTH_PASSWORD ="master2025"
DATABASE =  "llmagraphtrkg"

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_PROP = "embedding"

TOP_K = int(os.getenv("KNN_TOPK", "15"))
SIM_CUTOFF = float(os.getenv("KNN_SIM_CUTOFF", "0.80")) 
REL_TYPE = "SIMILAR"
REL_SCORE_PROP = "score"

G_EMBED = "g_entities_embed"
G_SIM = "g_entities_sim"


FETCH_BATCH = int(os.getenv("EMBED_FETCH_BATCH", "200"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "100"))


def build_entity_text(row: Dict[str, Any]) -> str:

    parts = []
    if row.get("id"):
        parts.append(str(row["id"]))
    if row.get("entityType"):
        parts.append(f"Type: {row['entityType']}")
    if row.get("description"):
        parts.append(f"Description: {row['description']}")
    return " | ".join(parts)


def chunk_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def main():
    # --- Neo4j driver (for reading/writing embeddings) ---
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

   
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)


    with driver.session(database=DATABASE) as session:
        session.run("MATCH ()-[r:SIMILAR]-() DELETE r;")
        session.run("MATCH (e:Entity) REMOVE e.communities;")
    print("Old SIMILAR rels deleted, communities removed.")

 

    total_updated = 0

    while True:
        with driver.session(database=DATABASE) as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN elementId(e) AS eid,
                       e.id AS id,
                       e.entityType AS entityType,
                       e.description AS description
                LIMIT $limit
                """,
                limit=FETCH_BATCH
            ).data()

        if not rows:
            break

        texts = [build_entity_text(r) for r in rows]

        # Embed in smaller batches
        all_vectors: List[List[float]] = []
        for batch_idx, text_batch in enumerate(chunk_list(texts, EMBED_BATCH), 1):
      
            
            if hasattr(embedder, "embed_documents"):
                vecs = embedder.embed_documents(text_batch)
            else:
                vecs = [embedder.embed_query(t) for t in text_batch]
            all_vectors.extend(vecs)

        # Write back
        payload = [{"eid": r["eid"], "vec": v} for r, v in zip(rows, all_vectors)]

        with driver.session(database=DATABASE) as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:Entity)
                WHERE elementId(e) = row.eid
                SET e.embedding = row.vec
                """,
                rows=payload
            )

        total_updated += len(rows)
        print(f"Embeddings written for {len(rows)} Entity nodes (total updated: {total_updated}).")

    print("Embedding step done.")

    # --- Step 2: GDS KNN + Leiden ---
    gds = GraphDataScience(
        URI,
        auth=(AUTH_USER, AUTH_PASSWORD),
        database=DATABASE,
    )

    # Drop in-memory graphs if exist
    if gds.graph.exists(G_EMBED)["exists"]:
        gds.graph.drop(G_EMBED)
    if gds.graph.exists(G_SIM)["exists"]:
        gds.graph.drop(G_SIM)

    # Project graph with node property embedding
    G1, meta1 = gds.graph.project(
        G_EMBED,
        "Entity",
        "*",
        nodeProperties=[EMBED_PROP],
    )
    print("Projected G1:", meta1)

    # KNN write -> SIMILAR relationships
    gds.knn.write(
        G1,
        nodeProperties=[EMBED_PROP],
        topK=TOP_K,
        similarityCutoff=SIM_CUTOFF,
        writeRelationshipType=REL_TYPE,
        writeProperty=REL_SCORE_PROP,
    )
    print(f"{REL_TYPE} relationships written with property '{REL_SCORE_PROP}'.")

    gds.graph.drop(G1)

    # Project SIMILAR-only undirected graph
    rel_proj = {
        REL_TYPE: {
            "type": REL_TYPE,
            "orientation": "UNDIRECTED",
            "properties": REL_SCORE_PROP,
        }
    }
    G2, meta2 = gds.graph.project(G_SIM, "Entity", rel_proj)
    print("Projected G2:", meta2)

    # Leiden -> list of communities per node
    gds.leiden.write(
        G2,
        writeProperty="communities",
        includeIntermediateCommunities=True,
        relationshipTypes=[REL_TYPE],
        relationshipWeightProperty=REL_SCORE_PROP,
    )
    print("Leiden communities written to :Entity.communities.")

    gds.graph.drop(G2)
    print("Done.")


if __name__ == "__main__":
    main()
