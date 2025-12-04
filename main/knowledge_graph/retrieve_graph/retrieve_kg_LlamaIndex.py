from dotenv import load_dotenv, find_dotenv
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv(find_dotenv())

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0)

username = "neo4j"
password = "master2025"
uri = "neo4j://127.0.0.1:7687"
database = "llamakg"

graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=uri,
    database=database,
)

# Index aus bestehendem GraphStore rekonstruieren
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
)
from llama_index.core.indices.property_graph import TextToCypherRetriever

DEFAULT_RESPONSE_TEMPLATE = (
    "Generated Cypher query:\n{query}\n\n"
    "Cypher Response:\n{response}"
)

DEFAULT_ALLOWED_FIELDS = ["text", "label", "type"]

# Template aus dem GraphStore holen
DEFAULT_TEXT_TO_CYPHER_TEMPLATE = index.property_graph_store.text_to_cypher_template

cypher_retriever = TextToCypherRetriever(
    index.property_graph_store,
    llm=llm,
    text_to_cypher_template=DEFAULT_TEXT_TO_CYPHER_TEMPLATE,
    response_template=DEFAULT_RESPONSE_TEMPLATE,
    cypher_validator=None,          # optional: kannst du später nutzen
    allowed_output_fields=DEFAULT_ALLOWED_FIELDS,
)
