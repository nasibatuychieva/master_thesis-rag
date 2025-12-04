from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.chat_models import ChatLlamaCpp

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from main.evaluation.logger import log_antwort


# ---------------------------------------------------------
# 1) LLM + Embeddings
# ---------------------------------------------------------

llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0,
    n_ctx=4096,
    max_tokens=1024,
    n_threads=4,
    n_gpu_layers=0,
)

embedding_provider = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2) Neo4j-Verbindung
# ---------------------------------------------------------

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
    database="neo4j",
)


# ---------------------------------------------------------
# 3) VectorStore aus bestehendem Index (nur Text)
# ---------------------------------------------------------

vector_store = Neo4jVector.from_existing_index(
    embedding=embedding_provider,
    graph=graph,
    index_name="chunkVector",            # dein existierender Index
    embedding_node_property="textEmbedding",
    text_node_property="text",
    node_label="Chunk",
)


# ---------------------------------------------------------
# 4) Hilfsfunktion: Nachbarn eines Chunk in Neo4j holen
# ---------------------------------------------------------

def get_neighbors_for_chunk(chunk_id: str, limit: int = 20):
    """
    Holt 1-Hop-Nachbarn eines Chunk-Knotens und gibt einfache Tripel zurück.
    """
    query = """
    MATCH (c:Chunk {id: $cid})-[:PART_OF]->(p:Product)
    OPTIONAL MATCH path = (p)-[r*1..2]-(n)
    RETURN DISTINCT
    p.id AS product_id,
    type(r) AS rel_type,
    labels(n) AS labels,
    n.id AS neighbor_id,
    coalesce(n.name, n.title, n.id) AS neighbor_name
    LIMIT 100

    """
    rows = graph.query(query, params={"cid": chunk_id, "limit": limit})
    triples = []
    for row in rows:
        rel = row["rel_type"]
        labels = row["labels"] or []
        target_label = labels[0] if labels else "Node"
        target_id = row["id"]
        target_name = row["label"]
        triples.append(
            f"(Chunk {chunk_id}) -[{rel}]-> ({target_label} {target_id}: {target_name})"
        )
    return triples


# ---------------------------------------------------------
# 5) Retrieval + Formatierung: Text + Graph-Kontext
# ---------------------------------------------------------

def retrieve_and_format(query: str, k_chunks: int = 4) -> str:
    """
    1. Vektor-Suche nach relevanten Chunks
    2. Für jeden Chunk 1-Hop-Nachbarn holen
    3. Alles in einen großen Kontext-String packen
    """
    docs = vector_store.similarity_search(query, k=k_chunks)

    if not docs:
        return "NO CONTEXT FOUND."

    parts = []
    for d in docs:
        md = d.metadata or {}

        # Je nachdem, wie du den Chunk gespeichert hast:
        # häufig 'id' oder 'chunk' – wir versuchen beides.
        chunk_id = md.get("id") or md.get("chunk") or md.get("node_id")

        graph_triples = []
        if chunk_id:
            graph_triples = get_neighbors_for_chunk(str(chunk_id), limit=20)

        part_lines = [
            f"Chunk ID: {chunk_id}",
            "Graph neighbors (1 hop):",
            *(graph_triples or ["(none)"]),
            "",
            "Chunk text:",
            d.page_content,
        ]
        parts.append("\n".join(part_lines))

    sep = "\n\n" + "-" * 60 + "\n\n"
    return sep + sep.join(parts)


# ---------------------------------------------------------
# 6) Prompt + Chain (LCEL)
# ---------------------------------------------------------

instructions = (
    "You are a helpful technical assistant for product documentation.\n"
    "Use the context from text chunks AND their connected graph nodes.\n"
    "If you are not sure, say that you don't know.\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ]
)

# LCEL-Chain: Frage -> (Retrieval+Format) -> Prompt -> LLM
rag_chain = (
    {
        "context": lambda q: retrieve_and_format(q),
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------
# 7) QA-Funktion + CLI
# ---------------------------------------------------------

def answer_question(question: str) -> str:
    return rag_chain.invoke(question)


if __name__ == "__main__":
    print("Hybrid RAG (Neo4jVector + Graph-Traversal via Cypher). 'exit' zum Beenden.\n")
    SCRIPT_NAME = "hybrid_manual_traversal.py"

    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        out = answer_question(q)
        print("\n" + out + "\n")
        log_antwort(SCRIPT_NAME, q, out)
