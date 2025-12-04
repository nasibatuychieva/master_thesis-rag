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
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# Connect to Neo4j database
# Connect to Neo4j database

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
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


chain = GraphCypherQAChain.from_llm(
  llm, graph=graph, verbose=True,
    allow_dangerous_requests=True,
    top_k = 2
)

chain.run("Which products belong to the Education?")