from neo4j import GraphDatabase

# -----------------------------
# FIXED CONFIG (NO ENV)
# -----------------------------
import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"   


CHUNK_LABEL = "Chunk"
ENTITY_LABEL = "__Entity__"       
MENTIONS_REL = "FROM_CHUNK"

EMBED_PROP = "embedding"


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        print("\n=== STEP 1: Graph Preconditions Check ===\n")

        # 1) counts
        q_counts = f"""
        RETURN
          count{{ (c:{CHUNK_LABEL}) }} AS chunks,
          count{{ (e:{ENTITY_LABEL}) }} AS entities,
          count{{ ()-[:{MENTIONS_REL}]-() }} AS mentions_rels
        """
        res = session.run(q_counts).single()
        print("[Counts]", dict(res))

        # 2) is MENTIONS direction correct? (Chunk -> entity)
        q_dir = f"""
        MATCH (c:{CHUNK_LABEL})-[:{MENTIONS_REL}]-(e:{ENTITY_LABEL})
        RETURN count(*) AS chunk_to_entity
        """
        dir_ok = session.run(q_dir).single()["chunk_to_entity"]
        print("[MENTIONS direction] Chunk -> entity:", dir_ok)

        # 3) embedding presence
        q_embed = f"""
        MATCH (e:{ENTITY_LABEL})
        RETURN
          count(e) AS total,
          count(CASE WHEN e.{EMBED_PROP} IS NOT NULL THEN 1 END) AS with_embedding
        """
        emb = session.run(q_embed).single()
        print("[Embeddings]", dict(emb))

        # 4) sample entity props
        q_sample = f"""
        MATCH (e:{ENTITY_LABEL})
        RETURN e.id AS id, e.name AS name, e.product AS product, e.product_category AS product_category
        LIMIT 5
        """
        sample = session.run(q_sample).data()
        print("[Sample entity rows]")
        for r in sample:
            print("  ", r)

        # 5) Check GDS installed (returns version if available)
        try:
            gds_ver = session.run("CALL gds.version() YIELD version RETURN version").single()["version"]
            print("[GDS] version:", gds_ver)
        except Exception as e:
            print("[GDS] NOT available or not permitted:", e)

    driver.close()
    print("\Finished.\n")


if __name__ == "__main__":
    main()
