from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from graphdatascience import GraphDataScience

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"  # prüfe den Namen

# ============================================================
# 1) Verbindung zu Neo4j
# ============================================================

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

# ============================================================
# 2) OPTIONAL: Embeddings erzeugen (falls 'embedding' noch nicht existiert)
# ============================================================

embedding_provider = OpenAIEmbeddings(model="text-embedding-3-small")

Neo4jVector.from_existing_graph(
    embedding=embedding_provider,
    graph=graph,
    node_label="Entity",
    text_node_properties=["id"],   # oder ["entityType", "id", "description"]
    embedding_node_property="embedding",
)

print("Embeddings for Entity nodes written (or updated).")

# ============================================================
# 3) GDS-Instanz
# ============================================================

gds = GraphDataScience(
    URI,
    auth=(AUTH_USER, AUTH_PASSWORD),
    database=DATABASE,
)

# ============================================================
# 4) Erster Graph: Embedding + beliebige Beziehungen
#    -> KNN schreibt SIMILAR-Kanten in die DB
# ============================================================

graph_name_1 = "entities_graph"
if gds.graph.exists(graph_name_1)["exists"]:
    gds.graph.drop(graph_name_1)
    print(f"Dropped existing in-memory graph '{graph_name_1}'.")

G1, result1 = gds.graph.project(
    graph_name_1,
    "Entity",
    "*",                   # vorhandene Beziehungen (Richtung egal)
    nodeProperties=["embedding"],
)

print("Projected Graph 1:", result1)

similarity_threshold = 0.95
top_k = 10

# KNN im WRITE-Mode -> schreibt :SIMILAR-Kanten in den Neo4j-Store
gds.knn.write(
    G1,
    nodeProperties=["embedding"],
    topK=top_k,
    similarityCutoff=similarity_threshold,
    writeRelationshipType="SIMILAR",
    writeProperty="score",
)

print("SIMILAR relationships written to Neo4j.")

# In-Memory-Graph 1 aufräumen
gds.graph.drop(G1)
print("In-memory graph 1 dropped.")

# ============================================================
# 5) Zweiter Graph: nur SIMILAR-Kanten, UNDIRECTED
# ============================================================

graph_name_2 = "communities_graph"
if gds.graph.exists(graph_name_2)["exists"]:
    gds.graph.drop(graph_name_2)
    print(f"Dropped existing in-memory graph '{graph_name_2}'.")

relationship_projection = {
    "SIMILAR": {
        "type": "SIMILAR",
        "orientation": "UNDIRECTED",
        "properties": "score",
    }
}

G2, result2 = gds.graph.project(
    graph_name_2,
    "Entity",
    relationship_projection,
)

print("Projected Graph 2 (SIMILAR only):", result2)

# Optional: WCC-Statistik
wcc_stats = gds.wcc.stats(G2)
print(f"Component count: {wcc_stats['componentCount']}")

# ============================================================
# 6) Leiden-Communities auf SIMILAR-Kanten berechnen
# ============================================================

gds.leiden.write(
    G2,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipTypes=["SIMILAR"],
    relationshipWeightProperty="score",
)

print("Leiden communities written to node property 'communities'.")

# In-Memory-Graph 2 aufräumen
gds.graph.drop(G2)
print("In-memory graph 2 dropped.")
