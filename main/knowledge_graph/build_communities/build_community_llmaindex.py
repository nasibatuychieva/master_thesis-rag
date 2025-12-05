from graphdatascience import GraphDataScience
from langchain_neo4j import Neo4jGraph
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmakg"

MAX_WORKERS = 10

# ============================================================
# 1) Neo4j-Graph + Embeddings für __Entity__
# ============================================================

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

gds = GraphDataScience(
    URI,
    auth=(AUTH_USER, AUTH_PASSWORD), database=DATABASE,
)
G, result = gds.graph.project(
    "entities",                   # Graph name
    "__Entity__",                 # Node projection
    "*",                          # Relationship projection
    nodeProperties=["embedding"]  # Configuration parameters
)
print(G, result)

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

gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
)