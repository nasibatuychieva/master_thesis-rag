import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.chat_models import ChatLlamaCpp
from neo4j import GraphDatabase

from neo4j_graphrag.retrievers import VectorCypherRetriever

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Connect to Neo4j database
# Connect to Neo4j database
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "testmaster123")
)


# Create embedder
llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0,
    n_ctx=4096,
    max_tokens=1024,
    n_threads=4,
    n_gpu_layers=0,
)

# embedder= HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


# Cypher examples as input/query pairs
examples = [
    """USER INPUT: 'Which products belong to the "Education"?' QUERY: MATCH (p:Product) MATCH (c:ProductCategory) WHERE c.id contains "Education" RETURN p,c"""
]

# Build the retriever

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    examples=examples,

)

# llm = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "Which products use USB Connector Usb-C® Port?"


response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()