from neo4j import GraphDatabase

# -----------------------------
# FIXED CONFIG (NO ENV)
# -----------------------------
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmakg"

ENTITY_LABEL = "entity"
COMM_LABEL = "__Community__"


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        print("\nRemoving old community nodes (if any)...")
        session.run(f"MATCH (c:{COMM_LABEL}) DETACH DELETE c;")

        print("Creating Community nodes + IN_COMMUNITY...")
        session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        WHERE e.communities IS NOT NULL

        UNWIND range(0, size(e.communities) - 1) AS level
        WITH e, level, toInteger(e.communities[level]) AS cid

        MERGE (c:{COMM_LABEL} {{
          level: level,
          communityId: cid
        }})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """)

        print(" Creating PARENT_COMMUNITY links deterministically...")
        session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        WHERE e.communities IS NOT NULL AND size(e.communities) >= 2
        UNWIND range(0, size(e.communities)-2) AS lvl
        WITH e,
             lvl,
             toInteger(e.communities[lvl])   AS childId,
             toInteger(e.communities[lvl+1]) AS parentId

        MATCH (child:{COMM_LABEL} {{level: lvl, communityId: childId}})
        MATCH (parent:{COMM_LABEL} {{level: lvl+1, communityId: parentId}})
        MERGE (child)-[:PARENT_COMMUNITY]->(parent);
        """)

    driver.close()
    print("\n Step 3 finished.\n")


if __name__ == "__main__":
    main()
