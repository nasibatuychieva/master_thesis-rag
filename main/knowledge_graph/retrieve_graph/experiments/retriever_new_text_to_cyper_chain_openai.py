"""
Fragen an den bestehenden Neo4j-Knowledge-Graph mit GraphCypherQAChain
unter Verwendung eines OpenAI-LLMs.

Voraussetzungen:
- Neo4j läuft auf bolt://localhost:7687
  User: neo4j / testmaster123
- Dein Graph enthält mindestens:
    (:ProductCategory {id, name})-[:HAS_PRODUCT]->(:Product {id, name})
    (:Product)-[:HAS_TUTORIAL]->(:Tutorial {id, name})
- In .env muss stehen: OPENAI_API_KEY=...
"""

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# 1) LLM: OpenAI (statt Ollama / ChatLlamaCpp)
# -------------------------------------------------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1",   # oder gpt-4.1, gpt-4o, gpt-4o-mini, etc.
    temperature=0,
)


# -------------------------------------------------------------------
# 2) Neo4j-Graph anbinden
# -------------------------------------------------------------------
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
    database="neo4j",
)

# -------------------------------------------------------------------
# 3) Kleines manuelles Schema (reduziert → besser für LLM)
# -------------------------------------------------------------------
CUSTOM_SCHEMA = """
(:Product {id, name})
(:ProductCategory {id, name})
(:Chunk {id, section})
(:Document {id, name})

(:ProductCategory)-[:HAS_PRODUCT]->(:Product)
(:Product)<-[:PART_OF]-(:Chunk)
(:Chunk)-[:IN_DOCUMENT]->(:Document)
"""


# -------------------------------------------------------------------
# 4) GraphCypherQAChain aufsetzen
# -------------------------------------------------------------------
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    enforce_schema=True,      # Nur konkrete Labels aus CUSTOM_SCHEMA erlauben
    # create_schema=False,      # Kein Auto-Schema-Scan (spart Tokens)
    # schema=CUSTOM_SCHEMA,     # Unser kompaktes Schema
    allow_dangerous_requests=True,
)


# -------------------------------------------------------------------
# 5) Hilfsfunktion für Abfragen
# -------------------------------------------------------------------
def ask(question: str):
    print(f"\n=== QUESTION ===\n{question}\n")
    answer = chain.invoke({"query": question})
    print("=== ANSWER ===")
    print(answer)
    print()


# -------------------------------------------------------------------
# 6) Beispiele
# -------------------------------------------------------------------
if __name__ == "__main__":
    ask("Which product is POWERED THROUGH Usb Power Input?")
    # Optional:
    # ask("Which products use USB Connector Usb-C® Port?")
    # ask("List all chunks for the Giga R1 board.")
