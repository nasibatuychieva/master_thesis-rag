# https://neo4j.com/blog/developer/global-graphrag-neo4j-langchain/
# Our process for __Entity__ resolution involves the following steps:

# Entities in the graph — Start with all entities within the graph.
# K-nearest graph — Construct a k-nearest neighbor graph, connecting similar entities based on text embeddings.
# Weakly Connected Components — Identify weakly connected components in the k-nearest graph, grouping entities that are 
# likely to be similar. 
# Add a word distance filtering step after these components have been identified.
# LLM evaluation — Use an LLM to evaluate these components and decide whether the entities within each component should be 
# merged, resulting in a final decision on __Entity__ resolution (for example, merging ‘Silicon Valley Bank’ and ‘
# Silicon_Valley_Bank’ while rejecting the merge for different dates like ‘September 16, 2023’ and ‘September 2, 2023’).
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graphdatascience import GraphDataScience
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from neo4j_graphrag.llm import OpenAILLM
import os
MAX_WORKERS = 10
NUM_ARTICLES = 2000
graph_documents = []

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "simplekg"

# Neo4j-Driver (DB wird über ENV oder default gewählt)


graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

#USE_OPENAI_EMBEDDINGS = True  # via .env oder config

#if USE_OPENAI_EMBEDDINGS:
from langchain_openai import OpenAIEmbeddings
embedding_provider = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
EMBED_DIM = 1536

# Wichtig: Label & Properties an deine Struktur anpassen!
vector = Neo4jVector.from_existing_graph(
    embedding=embedding_provider,
    graph=graph,
    node_label="__Entity__",                 # z.B. :__Entity__ oder :Element etc.
    text_node_properties=["id", "description"],
    embedding_node_property="embedding"  # wird als Property auf den Nodes gespeichert
)

print("Embeddings geschrieben.")

gds = GraphDataScience(
   URI, auth=(AUTH_USER, AUTH_PASSWORD), database=DATABASE
)

G, result = gds.graph.project(
    "communities",  #  Graph name
    "__Entity__",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

wcc = gds.wcc.stats(G)
print(f"Component count: {wcc['componentCount']}")
print(f"Component distribution: {wcc['componentDistribution']}")
# Component count: 1119
# Component distribution: {
#   "min":1,
#   "p5":1,
#   "max":9109,
#   "p999":43,
#   "p99":19,
#   "p1":1,
#   "p10":1,
#   "p90":7,
#   "p50":2,
#   "p25":1,
#   "p75":4,
#   "p95":10,
#   "mean":11.3 }

gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
)
graph.query("""
MATCH (e:`__Entity__`)
UNWIND range(0, size(e.communities) - 1 , 1) AS index
CALL {
  WITH e, index
  WITH e, index
  WHERE index = 0
  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET c.level = index
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
CALL {
  WITH e, index
  WITH e, index
  WHERE index > 0
  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET current.level = index
  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
  ON CREATE SET previous.level = index - 1
  MERGE (previous)-[:IN_COMMUNITY]->(current)
  RETURN count(*) AS count_1
}
RETURN count(*)
""")

graph.query("""
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:FROM_CHUNK]-(d:Chunk)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
""")


community_info = graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level IN [0,1,4]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
 whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId, 
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""")

community_template = """Based on the provided nodes and relationships that belong to the same graph community,
generate a natural language summary of the provided information:
{community_info}

Summary:"""  # noqa: E501

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input triples, generate the information summary. No pre-amble.",
        ),
        ("human", community_template),
    ]
)
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"temperature": 0}
)
community_chain = community_prompt | llm | StrOutputParser()

def prepare_string(data):
    nodes_str = "Nodes are:n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}n"

    rels_str = "Relationships are:n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}n"

    return nodes_str + "n" + rels_str

def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary}
summaries = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_community, community): community for community in community_info}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing communities"):
        summaries.append(future.result())
graph.query("""
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.summary = row.summary
""", params={"data": summaries})