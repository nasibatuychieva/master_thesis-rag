import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- Neo4j ---
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "testmaster123")
)

# --- lokales Embedding-Modell (HuggingFace, lokal geladen) ---
# embedder = SentenceTransformerEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )
embedder= HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2" )

# --- Cypher für zusätzliche Graph-Info ---
retrieval_query = """
WITH node AS c, score

// 1) Produkt über PART_OF
OPTIONAL MATCH (c)-[:PART_OF]->(p1:Product)

// 2) Produkt als Entity
OPTIONAL MATCH (c)-[:HAS_ENTITY]->(p2:Product)

// 3) p auswählen
WITH c, score, coalesce(p1, p2) AS p


// --- OPTION C: Produkt-Umfeld dynamisch einsammeln ---
// alle direkten Nachbarn des Produkts (egal welche Beziehung)
OPTIONAL MATCH (p)-[pr]-(p_neighbor)

// alle direkten Nachbarn des Chunks
OPTIONAL MATCH (c)-[cr]-(c_neighbor)

WITH
    c, score, p,
    collect(DISTINCT {rel: type(pr), node: p_neighbor.id}) AS product_neighbors,
    collect(DISTINCT {rel: type(cr), node: c_neighbor.id}) AS chunk_neighbors


// 4) Dokument optional einsammeln
OPTIONAL MATCH (c)-[:IN_DOCUMENT]->(d:Document)

RETURN
  c.text AS text,
  c.id   AS chunk_id,
  p.id   AS product_id,
  d.id   AS document_id,
  score  AS similarityScore,
  product_neighbors AS product_context,
  chunk_neighbors AS chunk_context

"""

retriever = VectorCypherRetriever(
    driver,
    index_name="chunkVector",           # dein bestehender Vector-Index
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# embedding_node_property="textEmbedding",
#     text_node_property="text",
#     node_label="Chunk",

llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0,
    n_ctx=4096,
    max_tokens=1024,
    n_threads=4,
    n_gpu_layers=0,
)
from langchain_core.runnables import RunnablePassthrough

def build_context(retriever_result):
    blocks = []
    for i, item in enumerate(retriever_result.items, start=1):
        md = item.metadata or {}
        text = item.content or ""
        chunk_id    = md.get("chunk_id")
        product_id  = md.get("product_id")
        doc_id      = md.get("document_id")
        score       = md.get("similarityScore")

        block = [
            f"Result {i}",
            f"  Score:      {score}",
            f"  Chunk ID:   {chunk_id}",
            f"  Product ID: {product_id}",
            f"  Document:   {doc_id}",
            "",
            text,
        ]
        blocks.append("\n".join(block))

    if not blocks:
        return "NO CONTEXT FOUND."
    return "\n\n" + ("\n\n" + "-"*60 + "\n\n").join(blocks)
instructions = (
    "You are a technical assistant for Arduino products.\n"
    "Use ONLY the provided context to answer the question.\n"
    "Always mention which product(s) and chunk IDs your answer is based on.\n"
    "If the context is insufficient, say that you don't know."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ]
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)
def answer_with_vector_cypher(question: str, top_k: int = 2) -> str:
    # 1) VectorCypherRetriever → GraphRAG-Retrieval
    retriever_result = retriever.search(
        query_text=question,
        top_k=top_k,
    )

    # 2) Kontextstring bauen
    context = build_context(retriever_result)

    # 3) Qwen mit LCEL aufrufen
    answer = chain.invoke({
        "context": context,
        "input": question,
    })

    # Optional: Quellen aus Metadaten extrahieren
    doc_names = {
        (item.metadata or {}).get("document_id")
        for item in retriever_result.items
        if item.metadata and item.metadata.get("document_id")
    }

    if doc_names:
        answer += "\n\nSources:\n" + "\n".join(f"- {d}" for d in doc_names)

    return answer


if __name__ == "__main__":
    print("GraphRAG with VectorCypherRetriever + Qwen. Type 'exit' to quit.\n")
    while True:
        q = input("> ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        out = answer_with_vector_cypher(q, top_k=2)
        print("\n" + out + "\n")

    driver.close()
