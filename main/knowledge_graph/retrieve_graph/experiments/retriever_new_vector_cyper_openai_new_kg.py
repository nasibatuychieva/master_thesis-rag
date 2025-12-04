import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from main.evaluation.logger import log_antwort
# ----------------------------------------------------------
# 1) Neo4j-Verbindung
# ----------------------------------------------------------

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "testmaster123"

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))

# ----------------------------------------------------------
# 2) Embeddings & Retriever
# ----------------------------------------------------------

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
from langchain_community.embeddings import HuggingFaceEmbeddings
# embedder = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

retrieval_query = """
WITH node AS c, score

// Metadaten
## Product
OPTIONAL MATCH (c)-[:PART_OF]->(p:Product)

#Product Category
Optional MATCH (p:Product)-[:BELONGS_TO]->(pc:ProductCategory)

OPTIONAL MATCH (c)-[:IN_DOCUMENT]->(d:Document)

// Entities im Chunk
OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)

// Community-Hierarchie (z.B. feinste Ebene level = 2)
OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(comm:__Community__ {level: 2})

// andere Entities in derselben Community
OPTIONAL MATCH (comm)<-[:IN_COMMUNITY]-(e2:Entity)

// optional: Relation zwischen e und e2, falls vorhanden
OPTIONAL MATCH (e)-[r]-(e2)

WITH c, score, p, d,
     collect(DISTINCT {
       eLabel: labels(e)[0],
       eId: e.id,
       communityId: comm.id,
       e2Label: labels(e2)[0],
       e2Id: e2.id,
       relType: type(r)
     }) AS graph_triples

RETURN
  c.id AS chunk_id,
  score AS similarityScore,
  p.id AS product_id,
  d.id AS document_id,
  graph_triples

"""

retriever = VectorCypherRetriever(
    driver,
    index_name="chunkVector",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# ----------------------------------------------------------
# 3) LLM & GraphRAG-Pipeline
# ----------------------------------------------------------

llm = OpenAILLM(model_name="gpt-4o")

rag = GraphRAG(retriever=retriever, llm=llm)

def get_products_by_category(driver, category_name: str):
    cypher = """
    MATCH (pc:ProductCategory)
    WHERE toLower(pc.id) CONTAINS toLower($cat)
    MATCH (pc)-[:HAS_PRODUCT]->(p:Product)
    RETURN pc.id AS category, collect(DISTINCT p.id) AS products
    """
    with driver.session() as session:
        records = session.run(cypher, cat=category_name).data()

    if not records:
        return f"No category matching '{category_name}' was found."

    lines = []
    for row in records:
        cat = row["category"]
        prods = row["products"]
        lines.append(
            f"Category '{cat}' has {len(prods)} product(s):\n" +
            "\n".join(f"- {p}" for p in prods)
        )
    return "\n\n".join(lines)


# ----------------------------------------------------------
# 4) Interaktive CLI
# ----------------------------------------------------------

def chat_loop(top_k: int = 5):
    print("GraphRAG + OpenAI. Tippe deine Frage (oder 'exit' zum Beenden).\n")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        q_lower = question.lower()

        # Sonderfall: "Which products belong to Product Category X?"
        if "which products" in q_lower and "product category" in q_lower:
            marker = "product category"
            idx = q_lower.find(marker)
            if idx != -1:
                cat_name = question[idx + len(marker):].strip(" ?.!:")
                answer = get_products_by_category(driver, cat_name)

                print("\nAntwort (direkt aus dem Knowledge Graph):\n")
                print(answer)
                print("\n" + "-" * 60 + "\n")

                # 🔥 Logging für diesen Fall
                log_antwort("GraphRAG_KG_Query", question, answer)
                continue

        # Standard: GraphRAG über Chunks
        response = rag.search(
            query_text=question,
            retriever_config={"top_k": top_k},
            return_context=True,
        )
        answer = response.answer  # ✨ wichtig für Logging

        print("\nAntwort:\n")
        print(answer)

        # Optional: Quellen
        doc_ids = {
            (item.metadata or {}).get("document_id")
            for item in response.retriever_result.items
            if item.metadata
        }
        if doc_ids:
            print("\nQuellen (document_id):")
            for d in doc_ids:
                if d:
                    print(f"- {d}")

        print("\n" + "-" * 60 + "\n")

        # Logging für RAG Antwort
        log_antwort("GraphRAG_Vector_Query", question, answer)



if __name__ == "__main__":
    try:
        chat_loop(top_k=5)
    finally:
        driver.close()
