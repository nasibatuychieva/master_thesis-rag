

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())


from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "simplekg"

cyper = """
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (n:Chunk)
ON n.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};
"""

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

def run_query(query, params=None):
    with driver.session(database=DATABASE) as session:
        return list(session.run(query, params or {}))


run_query(cyper)

print("Vector Index created successfully.")


