# -------------------------------------------------------
# 1) LLM + Embeddings (lokal Qwen)
# -------------------------------------------------------
from llama_index.core import Document, KnowledgeGraphIndex, Settings, StorageContext
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = LlamaCPP(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0.0,
    context_window=4096,
    max_new_tokens=256,
    model_kwargs={"n_threads": 4, "n_gpu_layers": 0},
)

embed = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = llm
Settings.embed_model = embed

# -------------------------------------------------------
# 2) Test-Daten als Document-Objekte
# -------------------------------------------------------

texts = [
    "Sarah works at PrismaticAI as a hardware engineer. "
    "She specializes in embedded systems.",

    "Michael is a software developer. "
    "He also works for PrismaticAI and focuses on backend systems."
]

documents = [Document(text=t) for t in texts]

# -------------------------------------------------------
# 3) KnowledgeGraphIndex bauen
# -------------------------------------------------------

print("[DEBUG] Baue KnowledgeGraphIndex…")

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=5,
    include_embeddings=True,
)

storage_context = index.storage_context

# -------------------------------------------------------
# 4) GraphRAG-Retriever + QueryEngine
# -------------------------------------------------------

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    embed_model=embed,
    llm=llm,
    similarity_top_k=5,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=graph_rag_retriever,
    llm=llm,
)

# -------------------------------------------------------
# 5) Test-Funktion
# -------------------------------------------------------

def ask(q: str):
    print(f"\n=== Frage: {q}")
    resp = query_engine.query(q)
    print("Antwort:", resp)
    print("Source-Nodes:", getattr(resp, "source_nodes", None))

ask("Where does Sarah work?")
ask("Who works at PrismaticAI?")
ask("Does Michael work for the same company as Sarah?")
