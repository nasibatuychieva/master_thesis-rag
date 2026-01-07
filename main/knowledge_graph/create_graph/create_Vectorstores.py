

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())


from neo4j import GraphDatabase



from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
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
