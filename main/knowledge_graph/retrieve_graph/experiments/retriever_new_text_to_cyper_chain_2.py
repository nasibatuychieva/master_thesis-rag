"""
Fragen an den bestehenden Neo4j-Knowledge-Graph mit GraphCypherQAChain
und lokalem Qwen-Modell (llama.cpp / ChatLlamaCpp).

Voraussetzungen:
- neo4j läuft auf bolt://localhost:7687 mit User neo4j / testmaster123
- dein Graph enthält mindestens:
    (:ProductCategory {id, name})-[:HAS_PRODUCT]->(:Product {id, name})
    (:Product)-[:HAS_TUTORIAL]->(:Tutorial {id, name})
- Modell-Datei: C:\models\qwen2.5-7b-instruct-q3_k_m.gguf
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_community.chat_models import ChatLlamaCpp


# -------------------------------------------------------------------
# 1) LLM: lokales Qwen-Modell über llama.cpp
# -------------------------------------------------------------------
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)



# -------------------------------------------------------------------
# 2) Neo4j-Graph anbinden
# -------------------------------------------------------------------
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
    database="neo4j",      # ggf. anpassen, wenn du eine andere DB nutzt
)

# *** WICHTIG: Schema klein halten, um den Prompt kurz zu halten ***
# Wir definieren ein minimales Schema manuell, das zu deinem Setup passt.
# Passe es an, falls deine Labels/Relationen anders heißen.
CUSTOM_SCHEMA = """
(:Product {id, name})
(:ProductCategory {id, name})
(:Chunk {id, section})
(:Document {id, name})

(:ProductCategory)-[:HAS_PRODUCT]->(:Product)
(:Product)<-[:PART_OF]-(:Chunk)
(:Chunk)-[:IN_DOCUMENT]->(:Document)
"""

# Wenn du später mehr brauchst, kannst du das Schema erweitern,
# aber halte es so knapp wie möglich.


# -------------------------------------------------------------------
# 3) GraphCypherQAChain aufsetzen
# -------------------------------------------------------------------
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    enforce_schema=True,
    create_schema=False,
    schema=CUSTOM_SCHEMA,
    allow_dangerous_requests=True,
)


def ask(question: str) -> None:
    """Hilfsfunktion für interaktive Fragen."""
    print(f"\n=== QUESTION ===\n{question}\n")
    answer = chain.invoke({"query": question})
    print("=== ANSWER ===")
    print(answer)
    print()


if __name__ == "__main__":
    # Beispiel-Frage aus deinem Kontext
    ask("Which products belong to the Education?")

    # Weitere Testfragen kannst du hier ergänzen, z.B.:
    # ask("Which products use USB Connector Usb-C® Port?")
