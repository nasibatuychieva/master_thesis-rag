from graphdatascience import GraphDataScience
import os

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "testmaster123"

gds = GraphDataScience(
    URI,
    auth=(AUTH_USER, AUTH_PASSWORD ),
)
##  Leiden algorithm
G, result = gds.graph.project(
    "communities",  #  Graph name
    "Entity",  #  Node projection
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