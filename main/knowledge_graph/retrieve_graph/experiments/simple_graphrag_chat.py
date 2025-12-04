from dotenv import load_dotenv
load_dotenv()

import os
import re

# --- LlamaIndex-Basics ---
from llama_index.core import Settings, StorageContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Neo4j + Graph Store ---
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# --- GraphRAG Retriever + QueryEngine ---
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


# ---------------------------------------------------------------------------
# 1) LLM + Embeddings
# ---------------------------------------------------------------------------

llm = LlamaCPP(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0.0,
    context_window=4096,
    max_new_tokens=256,   # kleiner, damit es schneller geht
    model_kwargs={
        "n_threads": 4,
        "n_gpu_layers": 0,
    },
)

embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# global für LlamaIndex setzen
Settings.llm = llm
Settings.embed_model = embedding_model


# ---------------------------------------------------------------------------
# 2) Neo4j Graph Store
# ---------------------------------------------------------------------------

graph_store = Neo4jGraphStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

print("[DEBUG] Knoten im Graph:",
      graph_store.query("MATCH (n) RETURN count(n) AS c"))
print("[DEBUG] Relationen im Graph:",
      graph_store.query("MATCH ()-[r]->() RETURN count(r) AS c"))

print("[DEBUG] Index-Structs im StorageContext:")
print(storage_context.index_store.index_structs)


# ---------------------------------------------------------------------------
# 3) KnowledgeGraphRAGRetriever + QueryEngine
#    (ja, deprecated – aber erstmal egal, Hauptsache es läuft)
# ---------------------------------------------------------------------------

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    llm=llm,
    embed_model=embedding_model,
    similarity_top_k=5,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=graph_rag_retriever,
    llm=llm,
)


# ---------------------------------------------------------------------------
# 4) Optional: Kategorie-Erkennung (wie bei dir)
# ---------------------------------------------------------------------------

# def extract_category_from_question(question: str) -> str | None:
#     q = question.strip().rstrip("?.!").lower()

#     patterns_en = [
#         r"which products(?:\s+do)?\s+belong to\s+(.*)",
#         r"what products(?:\s+do)?\s+belong to\s+(.*)",
#         r"which products are in\s+(.*)",
#         r"list (?:all )?products in\s+(.*)",
#     ]

#     patterns_de = [
#         r"welche produkte\s+gehören\s+zur\s+(.*)",
#         r"welche produkte\s+gehören\s+zu\s+(.*)",
#         r"welche produkte\s+gibt es in\s+(.*)",
#     ]

#     for pat in patterns_en + patterns_de:
#         m = re.match(pat, q)
#         if m:
#             cat = m.group(1).strip()
#             cat = re.sub(r"\b(family|familie|category|kategorie)\b$", "", cat).strip()
#             return cat if cat else None

#     return None


# def answer_products_by_category(category_query: str) -> str:
#     """Direkter Neo4j-Query über graph_store (OHNE LangChain)."""
#     records = graph_store.query(
#         """
#         MATCH (pc:ProductCategory)
#         WHERE toLower(pc.id) CONTAINS toLower($cat)
#         MATCH (pc)-[:HAS_PRODUCT]->(p:Product)
#         RETURN pc.id AS category, collect(DISTINCT p.id) AS products
#         """,
#         {"cat": category_query},
#     )

#     if not records:
#         return (
#             f"Ich konnte keine Produktkategorie zu '{category_query}' finden. "
#             f"Vielleicht heißt sie im Graphen anders."
#         )

#     lines = []
#     for row in records:
#         cat = row["category"]
#         products = row["products"] or []
#         if not products:
#             lines.append(f"Kategorie '{cat}' hat keine Produkte im Graphen.")
#         else:
#             header = f"Kategorie '{cat}' hat {len(products)} Produkt(e):"
#             prod_lines = "\n".join(f"- {p}" for p in products)
#             lines.append(header + "\n" + prod_lines)

#     return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 5) Zentrale QA-Funktion
# ---------------------------------------------------------------------------

def answer_question(question: str) -> str:
    print("[DEBUG] Frage:", question)

    # category = extract_category_from_question(question)
    # print("[DEBUG] erkannte Kategorie:", category)

    # if category:
    #     print("[DEBUG] → direkte Kategorie-Abfrage in Neo4j")
    #     return answer_products_by_category(category)

    print("[DEBUG] → GraphRAG über query_engine.query()")
    try:
        response = query_engine.query(question)
    except Exception as e:
        print("[DEBUG] Fehler in query_engine:", repr(e))
        print("[DEBUG] → Fallback: nur LLM ohne Graph")
        return str(llm.complete(question))

    answer_text = getattr(response, "response", str(response))
    if not answer_text:
        answer_text = "Ich konnte keine sinnvolle Antwort finden."
    
    response = query_engine.query(question)
    print("[DEBUG] raw response:", repr(response))
    print("[DEBUG] source_nodes:", getattr(response, "source_nodes", None))


    return answer_text

test_query = "Which products use USB Connector Usb-C® Port?"
nodes = graph_rag_retriever.retrieve(test_query)
print("[DEBUG] retrieve() Ergebnis:", nodes)
print("[DEBUG] Anzahl gefundener Knoten:", len(nodes))
# ---------------------------------------------------------------------------
# 6) Einfache CLI-Schleife
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GraphRAG-Chat (Qwen + Neo4j). Tippe 'exit' zum Beenden.\n")

    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        out = answer_question(q)
        print("\n" + out + "\n")
