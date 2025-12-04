from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from main.evaluation.logger import log_antwort
# ---------------------------------------------------------------------------
# 1) LLM und Embeddings
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2) Neo4j-Verbindung
# ---------------------------------------------------------------------------
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
)

# ---------------------------------------------------------------------------
# 3) Reiner VectorStore (KEIN KG-Join, nur Chunk-Knoten)
#    Hinweis: Passe Property-Namen an dein Schema an (text, id, textEmbedding).
# ---------------------------------------------------------------------------
# Minimaler Retrieval-Query: nur Text + einfache Metadaten vom Chunk
retrieval_query = """
RETURN
  node.text  AS text,
  score      AS score,
  {chunk: node.id} AS metadata
"""

# Vorhandenen Vektorindex auf :Chunk nutzen
chunk_vector = Neo4jVector.from_existing_index(
    embedding=embedding_provider,
    graph=graph,
    index_name="chunkVector",            # Name wie in Neo4j angelegt
    embedding_node_property="textEmbedding",
    text_node_property="text",
    node_label="Chunk",
    retrieval_query=retrieval_query,     # KEINE Beziehungen, nur Chunk
)

retriever = chunk_vector.as_retriever(search_kwargs={"k": 4})

# ---------------------------------------------------------------------------
# 4) Prompt + Kontextformatierung (nur Chunks)
# ---------------------------------------------------------------------------
instructions = (
    "Answer the user's question using only the provided context from document chunks. "
    "Cite the chunk id when helpful. If the answer is not in the context, say you don't know."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    parts = []
    for d in docs:
        md = d.metadata or {}
        chunk_id = md.get("chunk")
        part = [
            f"Chunk ID: {chunk_id}",
            "",
            f"Text:\n{d.page_content}",
        ]
        parts.append("\n".join(part))
    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(parts)

# ---------------------------------------------------------------------------
# 5) RAG-Chain (nur: Retriever -> Prompt -> LLM)
# ---------------------------------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def find_chunk(question: str) -> str:
    return rag_chain.invoke(question)

# ---------------------------------------------------------------------------
# 6) Einfaches CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Pure Vector RAG (Chunks).Enter 'exit' for finishing.\n")
    
    SCRIPT_NAME = "retriever_rag.py" 
    while (q := input("> ")).strip().lower() != "exit":
        if not q:
            continue
        answer = find_chunk(q)
        print("\n" + answer + "\n")
        # log the answer
        log_antwort(SCRIPT_NAME, q, answer)

