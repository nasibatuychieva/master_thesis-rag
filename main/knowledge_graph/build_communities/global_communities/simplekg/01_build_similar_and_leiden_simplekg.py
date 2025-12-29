from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

# -----------------------------
# FIXED CONFIG (NO ENV)
# -----------------------------
import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"

ENTITY_LABEL = "__Entity__"
EMBED_PROP = "embedding"

TOP_K = 15
SIM_CUTOFF = 0.80
REL_TYPE = "SIMILAR"
REL_SCORE_PROP = "score"

G_EMBED = "g_simplekg_entities_embed"
G_SIM = "g_simplekg_entities_sim"


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        print("\n=== STEP 2: KNN + Leiden on __Entity__ ===\n")

        print("Cleanup old SIMILAR rels + old communities property...")
        session.run(f"MATCH ()-[r:{REL_TYPE}]-() DELETE r;")
        session.run(f"MATCH (e:{ENTITY_LABEL}) REMOVE e.communities;")

        emb = session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        RETURN count(e) AS total,
               count(CASE WHEN e.{EMBED_PROP} IS NOT NULL THEN 1 END) AS with_embedding
        """).single()

        total, with_emb = emb["total"], emb["with_embedding"]
        print(f"[Embeddings] total={total}, with_embedding={with_emb}")
        if with_emb == 0:
            raise RuntimeError("No embeddings found on __Entity__. Run STEP 1 first.")

    gds = GraphDataScience(URI, auth=(AUTH_USER, AUTH_PASSWORD), database=DATABASE)

    # drop in-memory graphs if exist
    if gds.graph.exists(G_EMBED)["exists"]:
        gds.graph.drop(G_EMBED)
    if gds.graph.exists(G_SIM)["exists"]:
        gds.graph.drop(G_SIM)

    print("\nProjecting in-memory graph with embeddings...")
    G1, meta1 = gds.graph.project(
        G_EMBED,
        ENTITY_LABEL,
        "*",
        nodeProperties=[EMBED_PROP],
    )
    print("Projected G1:", meta1)

    print("\nRunning KNN and writing SIMILAR relationships...")
    gds.knn.write(
        G1,
        nodeProperties=[EMBED_PROP],
        topK=TOP_K,
        similarityCutoff=SIM_CUTOFF,
        writeRelationshipType=REL_TYPE,
        writeProperty=REL_SCORE_PROP,
    )
    print(f"{REL_TYPE} relationships written.")

    gds.graph.drop(G1)

    print("\nProjecting SIMILAR-only graph for Leiden...")
    rel_proj = {
        REL_TYPE: {
            "type": REL_TYPE,
            "orientation": "UNDIRECTED",
            "properties": REL_SCORE_PROP,
        }
    }
    G2, meta2 = gds.graph.project(G_SIM, ENTITY_LABEL, rel_proj)
    print("Projected G2:", meta2)

    print("\nLeiden community detection (hierarchical)...")
    gds.leiden.write(
        G2,
        writeProperty="communities",
        includeIntermediateCommunities=True,
        relationshipTypes=[REL_TYPE],
        relationshipWeightProperty=REL_SCORE_PROP,
    )
    print("Leiden communities written to __Entity__.communities")

    gds.graph.drop(G2)
    driver.close()

    print("\Step 2 finished.\n")
    print("NOW CHECK:")
    print("1) MATCH ()-[r:SIMILAR]-() RETURN count(r);")
    print("2) MATCH (e:__Entity__) WHERE e.communities IS NOT NULL RETURN size(e.communities) AS levels, count(*) AS cnt ORDER BY cnt DESC;")

if __name__ == "__main__":
    main()
