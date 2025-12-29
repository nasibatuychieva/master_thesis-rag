import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE =  "llmagraphtrkg"


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:

        print("Removing old Community nodes (if any)...")
        session.run("""
            MATCH (c:__Community__)
            DETACH DELETE c
        """)

        print("Creating Community nodes + relationships...")

        session.run("""
        MATCH (e:Entity)
        WHERE e.communities IS NOT NULL

        UNWIND range(0, size(e.communities)-1) AS level
        WITH e, level, e.communities[level] AS cid

        MERGE (c:__Community {
            level: level,
            communityId: cid
        })

        MERGE (e)-[:IN_COMMUNITY {level: level}]->(c)
        """)

        print("Creating PARENT_COMMUNITY hierarchy...")

        session.run("""
        MATCH (c1:__Community)
        MATCH (c2:__Community)
        WHERE c1.level = c2.level - 1
          AND c1.communityId IN [
              cid IN collect {
                  MATCH (e:Entity)-[:IN_COMMUNITY]->(c1)
                  RETURN e.communities[c2.level]
              }
          ]
        MERGE (c1)-[:PARENT_COMMUNITY]->(c2)
        """)

    driver.close()
    print("Script 2 finished successfully.")


if __name__ == "__main__":
    main()
