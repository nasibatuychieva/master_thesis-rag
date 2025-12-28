from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "rag"

INDEX_NAME = "chunk_text_ft"

CREATE_INDEX_CYPHER = f"""
CREATE FULLTEXT INDEX {INDEX_NAME}
FOR (c:Chunk)
ON EACH [c.text]
"""

CHECK_INDEX_CYPHER = """
SHOW FULLTEXT INDEXES
YIELD name
WHERE name = $index_name
RETURN name
"""

def main():
    driver = GraphDatabase.driver(
        URI,
        auth=(AUTH_USER, AUTH_PASSWORD),
    )

    with driver.session(database=DATABASE) as session:
        # 1) Check if index already exists
        result = session.run(
            CHECK_INDEX_CYPHER,
            index_name=INDEX_NAME,
        )
        exists = result.single() is not None

        if exists:
            print(f"[OK] Fulltext index '{INDEX_NAME}' already exists.")
        else:
            print(f"[INFO] Creating fulltext index '{INDEX_NAME}' ...")
            session.run(CREATE_INDEX_CYPHER)
            print(f"[OK] Fulltext index '{INDEX_NAME}' created.")

    driver.close()

if __name__ == "__main__":
    main()
