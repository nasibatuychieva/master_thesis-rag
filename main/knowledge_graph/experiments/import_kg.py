from neo4j import GraphDatabase
import pandas as pd
import json

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_triplets(self, triplets, node_label="Entity"):
        with self.driver.session() as session:
            for triplet in triplets:
                head = triplet.get("head")
                relation = triplet.get("type").replace(" ", "_").upper()  # Für Cypher kompatibel
                tail = triplet.get("tail")
                
                if head and relation and tail:
                    session.run(
                        f"""
                        MERGE (h:{node_label} {{name: $head}})
                        MERGE (t:{node_label} {{name: $tail}})
                        MERGE (h)-[r:{relation}]->(t)
                        """,
                        head=head, tail=tail
                    )
        print(f" {len(triplets)} Tripel erfolgreich importiert.")

# Connect to DB 
importer = Neo4jImporter(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="testmaster123" #"testpass" 
)

# Load Tripel-Data
triplets = pd.read_json("entities_relations.json")
importer.import_triplets(triplets.to_dict(orient="records"))

# Close the connection
importer.close()
