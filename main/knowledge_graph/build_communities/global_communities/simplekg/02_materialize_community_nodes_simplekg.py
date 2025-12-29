from neo4j import GraphDatabase

# -----------------------------
# FIXED CONFIG (NO ENV)
# -----------------------------
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"

ENTITY_LABEL = "__Entity__"
COMMUNITY_LABEL = "__Community__"

IN_COMMUNITY = "IN_COMMUNITY"
PARENT_COMMUNITY = "PARENT_COMMUNITY"


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        print("\n=== STEP 3: Materialize __Community__ nodes ===\n")

        # constraints for uniqueness
        session.run(f"""
        CREATE CONSTRAINT community_key IF NOT EXISTS
        FOR (c:{COMMUNITY_LABEL})
        REQUIRE (c.level, c.communityId) IS UNIQUE;
        """)

        # cleanup previous materialization
        print("[Cleanup] Removing old community graph (if any)...")
        session.run(f"MATCH ()-[r:{IN_COMMUNITY}]->() DELETE r;")
        session.run(f"MATCH ()-[r:{PARENT_COMMUNITY}]->() DELETE r;")
        session.run(f"MATCH (c:{COMMUNITY_LABEL}) DELETE c;")
        print("[Cleanup] Done.\n")

        # create community nodes + IN_COMMUNITY edges for every level
        print("[Build] Creating __Community__ nodes and IN_COMMUNITY edges...")
        session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        WHERE e.communities IS NOT NULL AND size(e.communities) > 0
        UNWIND range(0, size(e.communities)-1) AS lvl
        WITH e, lvl, e.communities[lvl] AS cid
        MERGE (c:{COMMUNITY_LABEL} {{level: lvl, communityId: toInteger(cid)}})
        MERGE (e)-[:{IN_COMMUNITY}]->(c)
        """)
        print("[Build] Done.\n")

        # build parent links: lvl i -> lvl i+1
        print("[Build] Creating PARENT_COMMUNITY hierarchy...")
        session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        WHERE e.communities IS NOT NULL AND size(e.communities) > 1
        WITH e, e.communities AS comms
        UNWIND range(0, size(comms)-2) AS lvl
        WITH lvl,
             toInteger(comms[lvl]) AS childId,
             toInteger(comms[lvl+1]) AS parentId
        MERGE (child:{COMMUNITY_LABEL} {{level: lvl, communityId: childId}})
        MERGE (parent:{COMMUNITY_LABEL} {{level: lvl+1, communityId: parentId}})
        MERGE (child)-[:{PARENT_COMMUNITY}]->(parent)
        """)
        print("[Build] Done.\n")

    driver.close()
    print("Finished.\n")


if __name__ == "__main__":
    main()
