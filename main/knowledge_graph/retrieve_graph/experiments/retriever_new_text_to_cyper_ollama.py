from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OllamaLLM   # <- LLMInterface-Implementierung, lokal & kostenlos
from neo4j_graphrag.generation import GraphRAG

# ---------------------------------------------------------------------
# 1) Neo4j-Verbindung
# ---------------------------------------------------------------------
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "testmaster123")

driver = GraphDatabase.driver(URI, auth=AUTH)

# ---------------------------------------------------------------------
# 2) LLM: lokales Modell über Ollama
#    (du kannst hier auch ein anderes Ollama-Modell wählen, z.B. "qwen2.5:7b")
# ---------------------------------------------------------------------
llm = OllamaLLM(
    model_name="qwen3:4b",
    # model_params={"options": {"temperature": 0}},  # optional: kühlere Antworten
)

# ---------------------------------------------------------------------
# 3) Beispiele für NL->Cypher (helfen dem Modell beim Lernen)
#    Wichtig: hier deine eigenen Beispiele mit deinen Labels / Properties eintragen!
# ---------------------------------------------------------------------
examples = [
    # Beispiel 1: Produkte nach Kategorie
    """
    USER INPUT: 'Which products belong to the Education category?'
    QUERY: MATCH (pc:ProductCategory)<-[:HAS_PRODUCT]-(p:Product)
           WHERE toLower(pc.id) CONTAINS 'education'
           RETURN p, pc
    """,

    # Beispiel 2: Produkte mit USB-C Connector
    """
    USER INPUT: 'Which products use USB Connector Usb-C® Port?'
    QUERY: MATCH (p:Product)-[:HAS_ELEMENT]->(e:Element)
           WHERE toLower(e.id) CONTAINS 'usb-c'
              OR toLower(e.name) CONTAINS 'usb connector usb-c'
           RETURN p, e
    """,
]

# ---------------------------------------------------------------------
# 4) Text2CypherRetriever
#    -> er nutzt das LLM, um aus der Frage einen Cypher-Query zu bauen,
#       führt diesen Query in Neo4j aus und gibt die Records an GraphRAG.
# ---------------------------------------------------------------------
retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    examples=examples,      # hilft dem Modell, gute Cypher zu bauen
    # neo4j_schema=None,    # optional: Schema-String, sonst wird es aus der DB gelesen
)

# ---------------------------------------------------------------------
# 5) GraphRAG-Pipeline
# ---------------------------------------------------------------------
rag = GraphRAG(
    retriever=retriever,
    llm=llm,    # dasselbe LLM formuliert die finale Antwort
)

# ---------------------------------------------------------------------
# 6) Testfrage
# ---------------------------------------------------------------------
query_text = "What distinguishes the Arduino Nano 33 BLE from the Arduino Nano 33 BLE Sense?"

response = rag.search(
    query_text=query_text,
    return_context=True,   # Kontext / Records mit zurückgeben
)

print("ANSWER:")
print(response.answer)
print("\nCYPHER:")
print(response.retriever_result.metadata.get("cypher"))
print("\nCONTEXT RECORDS:")
for item in response.retriever_result.items:
    print(item)

driver.close()
