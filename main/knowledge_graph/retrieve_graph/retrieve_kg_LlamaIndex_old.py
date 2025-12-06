from dotenv import load_dotenv, find_dotenv
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PGRetriever,
)
from main.evaluation.logger import log_antwort   # falls vorhanden

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0)

username = "neo4j"
password = "master2025"
uri = "neo4j://127.0.0.1:7687"
database = "llmakg"          # <-- exakt der Name deiner DB!

# GraphStore an Neo4j anbinden
graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=uri,
    database=database,
)

# PropertyGraphIndex aus bestehendem Graphen rekonstruieren
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
)

# ---------------------------------------------------------------------------
# 2) Retriever definieren: Vector + Synonym
# ---------------------------------------------------------------------------

# nutzt Embeddings der Knoten (Kontext um relevante Knoten herum)
vector_retriever = VectorContextRetriever(
    graph_store=index.property_graph_store,   # wichtig!
    embed_model=embed_model,
    similarity_top_k=10,                      # wie viele ähnliche Knoten
)

# nutzt LLM, um semantisch ähnliche Knoten / Pfade zu finden
synonym_retriever = LLMSynonymRetriever(
    graph_store=index.property_graph_store,
    llm=llm,
)

# Kombi-Retriever: erst Sub-Retriever, dann LLM zum Zusammenführen
pg_retriever = PGRetriever(
    sub_retrievers=[synonym_retriever, vector_retriever],
    llm=llm,
)

# ---------------------------------------------------------------------------
# 3) Einfache Chat-Loop
# ---------------------------------------------------------------------------

# Query Engine aus dem PGRetriever bauen
def chat_loop():
    print("KG-PGRetriever (Synonym + Vector). Type your question or 'exit' to quit.\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        # 1) Retrieve KG context
        results = pg_retriever.retrieve(question)

        # 2) Kontext zusammensetzen
        context = ""
        for r in results:
            try:
                context += f"- {r.get_content()}\n"
            except:
                context += f"- {str(r)}\n"

        # 3) LLM generiert Antwort
        prompt = f"""
You are an expert in Arduino hardware and embedded systems.
Answer the user question using ONLY the following retrieved KG context.

Context:
{context}

Question: {question}

Final answer:
"""
        answer = llm.complete(prompt)

        print("\nAntwort:\n")
        print(answer.text)

        # Optional Logging
        try:
            log_antwort("KG_PGRetriever_Synonym_Vector", question, answer.text)
        except:
            pass


if __name__ == "__main__":
    chat_loop()
