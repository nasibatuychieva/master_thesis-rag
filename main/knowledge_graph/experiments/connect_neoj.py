from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "testmaster123"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Test
with driver.session() as session:
    result = session.run("RETURN 'Neo4j connected' AS msg")
    print(result.single()["msg"])
