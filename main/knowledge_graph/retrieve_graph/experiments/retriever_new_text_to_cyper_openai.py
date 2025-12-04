from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI  
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM           # <- OpenAI LLM Wrapper
from neo4j_graphrag.generation import GraphRAG
import os
import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever
# ---------------------------------------------------------------------
# 1) Neo4j-Verbindung
# ---------------------------------------------------------------------
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "testmaster123")

driver = GraphDatabase.driver(URI, auth=AUTH)

URI = "neo4j://127.0.0.1:7687"
AUTH = ("master_neo4j", "testmaster123")
# Create Cypher LLM 
t2c_llm = OpenAILLM(
    model_name="gpt-4o", 
    model_params={"temperature": 0}
)
t2c_custom_prompt = """
You are an expert in translating natural-language questions into valid Neo4j Cypher.

GRAPH SCHEMA (only use these labels and relationships):
{schema}

IMPORTANT RULES:
- Use ONLY the labels and relationship types from the schema above.
- NEVER invent new labels like :Category or relationships like :BELONGS_TO
  if they are not listed in the schema.
- When the user refers to a *category* of a product (e.g. "Education category",
  "products in Education family", "products in this category"),
  interpret this as node :ProductCategory and the relationship
  (:ProductCategory)<-[:HAS_PRODUCT]-(:Product).
- When the user talks about connectors, pins, or hardware elements of a product,
  use (:Product)-[:HAS_ELEMENT]->(:Element).
- You may use synonyms and semantic understanding:
  - "category", "segment", "family", "product line" → :ProductCategory
  - "belongs to", "in", "under", "part of" → the HAS_PRODUCT relation
- If something cannot be answered with the given schema, write a Cypher query
  that returns no rows but is still syntactically valid.

EXAMPLES (follow their style and structure):
{examples}

Now generate ONE Cypher query that answers the user question.

User question:
{query_text}

Return ONLY the Cypher query, no explanation, no backticks.
"""


# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    custom_prompt =t2c_custom_prompt
)

llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "What distinguishes the Arduino Nano 33 BLE from the Arduino Nano 33 BLE Sense?"
query_text = "Which products belong to the Education category?"

response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()