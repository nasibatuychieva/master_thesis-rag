

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())


from neo4j import GraphDatabase

# URI = "neo4j://127.0.0.1:7687"
# AUTH_USER = "neo4j"
# AUTH_PASSWORD = "master2025"
# DATABASE = "simplekg"

# cyper = """
# CREATE VECTOR INDEX chunkEmbedding_simplekg IF NOT EXISTS
# FOR (n:Chunk)
# ON n.embedding
# OPTIONS {indexConfig: {
#  `vector.dimensions`: 1536,
#  `vector.similarity_function`: 'cosine'
# }};
# """

# driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
# driver.verify_connectivity()

# def run_query(query, params=None):
#     with driver.session(database=DATABASE) as session:
#         return list(session.run(query, params or {}))


# run_query(cyper)

# print("Vector Index created successfully.")



# URI = "neo4j://127.0.0.1:7687"
# AUTH_USER = "neo4j"
# AUTH_PASSWORD = "master2025"
# DATABASE = "llmagraphtrkg"

# cyper = """
# CREATE VECTOR INDEX chunkEmbedding_llmagraphtrkg IF NOT EXISTS
# FOR (n:Chunk)
# ON n.embedding
# OPTIONS {indexConfig: {
#  `vector.dimensions`: 1536,
#  `vector.similarity_function`: 'cosine'
# }};
# """

# driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
# driver.verify_connectivity()

# def run_query(query, params=None):
#     with driver.session(database=DATABASE) as session:
#         return list(session.run(query, params or {}))


# run_query(cyper)

# print("Vector Index created successfully.")

# URI = "neo4j://127.0.0.1:7687"
# AUTH_USER = "neo4j"
# AUTH_PASSWORD = "master2025"
# DATABASE = "simplekg"

# cypher = """
# CREATE FULLTEXT INDEX chunkFulltext_simplekg IF NOT EXISTS
# FOR (n:Chunk)
# ON EACH [n.text];
# """

# driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
# driver.verify_connectivity()

# def run_query(query, params=None):
#     with driver.session(database=DATABASE) as session:
#         return list(session.run(query, params or {}))

# run_query(cypher)

# print("Fulltext Index created successfully.")


# URI = "neo4j://127.0.0.1:7687"
# AUTH_USER = "neo4j"
# AUTH_PASSWORD = "master2025"
# DATABASE = "llmagraphtrkg"

# cypher = """
# CREATE FULLTEXT INDEX chunkFulltext_llmagraphtrkg IF NOT EXISTS
# FOR (n:Chunk)
# ON EACH [n.text];
# """

# driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
# driver.verify_connectivity()

# def run_query(query, params=None):
#     with driver.session(database=DATABASE) as session:
#         return list(session.run(query, params or {}))

# run_query(cypher)

# print("Fulltext Index created successfully.")

from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_chunks():
    with driver.session(database=DATABASE) as session:
        result = session.run("MATCH (c:Chunk) RETURN id(c) AS id, c.text AS text")
        return [record.data() for record in result]

def set_embedding(node_id, vector):
    with driver.session(database=DATABASE) as session:
        session.run(
            """
            MATCH (c:Chunk) 
            WHERE id(c) = $id 
            SET c.embedding = $emb
            """,
            {"id": node_id, "emb": vector}
        )

def main():
    chunks = get_chunks()
    print(f"Found {len(chunks)} chunks.")
    for row in chunks:
        text = row["text"] or ""
        vec = embeddings.embed_query(text)
        set_embedding(row["id"], vec)
    print("Embeddings set for all Chunk nodes.")

if __name__ == "__main__":
    main()
