import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM          # <-- WICHTIG: LLM-Wrapper von neo4j_graphrag

from langchain_openai import OpenAIEmbeddings     # Embeddings aus OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
# ----------------------------------------------------------
# 1) Neo4j-Verbindung
# ----------------------------------------------------------

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "testmaster123"),
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")


# ----------------------------------------------------------
# 2) OpenAI Embeddings
#    Achtung: Für bestes Retrieval sollte der Vector-Index in Neo4j
#    mit demselben Embedding-Modell erstellt worden sein.
# ----------------------------------------------------------

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# ----------------------------------------------------------
# 3) Deine Retrieval-Query (übernommen)
# ----------------------------------------------------------

# retrieval_query = """
# WITH node AS c, score

# OPTIONAL MATCH (c)-[:PART_OF]->(p:Product)
# OPTIONAL MATCH path_p = (p)-[:HAS_TUTORIAL|HAS_PRODUCT|HAS_CATEGORY*1..2]-(p_nei)
# WITH c, score, p, p_nei, relationships(path_p) AS p_rels

# OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
# OPTIONAL MATCH path_e = (e)-[*1..2]-(e_nei)
# WITH c, score, p, p_nei, p_rels, e, e_nei, relationships(path_e) AS e_rels

# UNWIND coalesce(p_rels, []) AS pr
# UNWIND coalesce(e_rels, []) AS er

# RETURN DISTINCT
#   c.id AS chunk_id,
#   score AS similarityScore,
#   p.id AS product_id,
#   type(pr) AS product_rel_type,
#   labels(p_nei) AS product_neighbor_labels,
#   p_nei.id AS product_neighbor_id,
#   coalesce(p_nei.name, p_nei.id) AS product_neighbor_name,
#   labels(e) AS entity_labels,
#   e.id AS entity_id,
#   type(er) AS entity_rel_type,
#   labels(e_nei) AS entity_neighbor_labels,
#   e_nei.id AS entity_neighbor_id,
#   coalesce(e_nei.name, e_nei.id) AS entity_neighbor_name

# """
retrieval_query = """
// 1) Produkt zum Chunk
MATCH (node)-[:PART_OF]->(p:Product)

// 2) Dokument
OPTIONAL MATCH (node)-[:IN_DOCUMENT]->(d:Document)

// 3) Tutorial
OPTIONAL MATCH (node)-[:IN_TUTORIAL]->(t:Tutorial)

// 4) Elements des Produkts
OPTIONAL MATCH (p)-[rel]->(el:Element)
WITH node, score, p, d,
     coalesce(t.id, node.tutorial) AS tutorial_id,
     collect(DISTINCT {element: el.id, rel_type: type(rel)}) AS elements

// 5) Entities
OPTIONAL MATCH (node)-[:HAS_ENTITY]->(e1)
OPTIONAL MATCH (e1)-[r]-(e2)
WHERE (node)-[:HAS_ENTITY]->(e2)

WITH node, score, p, d, tutorial_id, elements,
     collect(DISTINCT e1.id) AS ents,
     collect(DISTINCT apoc.text.join([
         labels(startNode(r))[0], coalesce(startNode(r).id,''),
         type(r),
         labels(endNode(r))[0], coalesce(endNode(r).id,'')
     ], ' ')) AS kg

RETURN
  node.text AS text,
  score,
  {
    chunk:   node.id,
    product: p.id,
    document: d.id,
    elements: elements,   // <--- JETZT wirklich drin
    entities: ents,
    kg: kg,
    tutorial: tutorial_id
  } AS metadata;
"""

# ----------------------------------------------------------
# 4) VectorCypherRetriever
# ----------------------------------------------------------

retriever = VectorCypherRetriever(
    driver,
    index_name="chunkVector",      # dein bestehender Vector-Index
    embedder=embedder,
    retrieval_query=retrieval_query,
)


# ----------------------------------------------------------
# 5) OpenAI LLM über neo4j_graphrag-Wrapper
#    (NICHT ChatOpenAI verwenden, sonst kommt der system_instruction-Fehler)
# ----------------------------------------------------------

llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
   # model_params={"temperature": 0.0},
)


# ----------------------------------------------------------
# 6) GraphRAG-Pipeline
# ----------------------------------------------------------

rag = GraphRAG(retriever=retriever, llm=llm)


def answer_with_vector_cypher(question: str, top_k: int = 3):
    """
    Stellt eine Frage an GraphRAG + VectorCypherRetriever
    und gibt Antwort + Kontext-Items zurück.
    """
    result = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )

    print("\n=== ANSWER ===\n")
    print(result.answer)

    # print("\n=== CONTEXT ITEMS ===")
    # for item in result.retriever_result.items:
    #     print(item)
    #     print("-" * 60)

    return result


# ----------------------------------------------------------
# 7) CLI – beliebige Fragen eingeben
# ----------------------------------------------------------

if __name__ == "__main__":
    print("GraphRAG + VectorCypherRetriever + OpenAI. Type 'exit' to quit.\n")

    try:
        while True:
            q = input("> ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue

            answer_with_vector_cypher(q, top_k=3)
            print()
    finally:
        driver.close()
