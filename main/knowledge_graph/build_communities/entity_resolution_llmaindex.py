# ============================================================
# 0) Imports & Grundkonfiguration
# ============================================================
from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector

from graphdatascience import GraphDataScience

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

# OpenAI-Embeddings
embedding_provider = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
EMBED_DIM = 1536

# Alle __Entity__-Knoten mit Embeddings ausstatten (Property: embedding)
# Falls bereits Embeddings vorhanden sind, werden sie überschrieben.
# vector = Neo4jVector.from_existing_graph(
#     embedding=embedding_provider,
#     graph=graph,
#     node_label="__Entity__",          # Dein Entity-Label
#     text_node_properties=["id"],      # Textbasis für Embedding
#     embedding_node_property="embedding"
# )

# print("Embeddings für __Entity__ geschrieben.")

from graphdatascience import GraphDataScience

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
similarity_threshold = 0.95

gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold
)
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)
word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity__`)
    WHERE size(e.id) > 3 // longer than 3 characters
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Add text distance
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                  OR node.id CONTAINS n.id | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // combine groups together if they share elements
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    WITH distinct(combinedResult) as combinedResult
    // extra filtering
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """, params={'distance': word_edit_distance})

for row in potential_duplicate_candidates:
    print("Candidate group:", row["combinedResult"])
    
system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


extraction_llm = ChatOpenAI(model_name="gpt-4o").with_structured_output(
    Disambiguate
)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            user_template,
        ),
    ]
)

extraction_chain = extraction_prompt | extraction_llm


def entity_resolution(entities: List[str]) -> Optional[List[List[str]]]:
    result = extraction_chain.invoke({"entities": entities})
    if not result.merge_entities:
        return None
    return [el.entities for el in result.merge_entities]

merged_entities = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submitting all tasks and creating a list of future objects
    futures = [
        executor.submit(entity_resolution, el['combinedResult'])
        for el in potential_duplicate_candidates
    ]

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing documents"
    ):
        to_merge = future.result()
        if to_merge:    
            merged_entities.extend(to_merge)

print(f"Number of merge groups: {len(merged_entities)}")
print("Merged entities example:", merged_entities[:5])
result= graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    description:'combine',
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": merged_entities})

print("Merge result from Neo4j:", result)

# # Kurzer Check, wie viele __Entity__ jetzt ein Embedding haben
# cnt = graph.query("""
# MATCH (e:__Entity__)
# RETURN
#   sum(CASE WHEN e.embedding IS NOT NULL THEN 1 ELSE 0 END) AS withEmbedding,
#   count(e) AS total
# """)[0]

# print(f"Entities mit Embedding: {cnt['withEmbedding']} von {cnt['total']}")

# # ============================================================
# # 2) GDS-Client + Projection-Graph per Cypher
# # ============================================================

# gds = GraphDataScience(
#     URI,
#     auth=(AUTH_USER, AUTH_PASSWORD),
# )
# gds.set_database(DATABASE)

# # ggf. alten Projection-Graph löschen
# gds.run_cypher("CALL gds.graph.drop('entities', false) YIELD graphName RETURN graphName;")

# # Node-Query: nur Entities mit nicht-NULL-Embedding,
# # embedding wird explizit als Property zurückgegeben.
# node_query = """
# MATCH (e:__Entity__)
# WHERE e.embedding IS NOT NULL
# RETURN id(e) AS id, e.embedding AS embedding
# """

# # Relationship-Query: hier erst mal **keine** Kanten projizieren.
# # (k-NN erzeugt später die SIMILAR-Kanten im GDS-Graphen.)
# relationship_query = """
# MATCH (e:__Entity__)
# WHERE false
# RETURN id(e) AS source, id(e) AS target
# """

# projection_result = gds.run_cypher(
#     """
#     CALL gds.graph.project.cypher(
#       'entities',
#       $nodeQuery,
#       $relQuery
#     )
#     YIELD graphName, nodeCount, relationshipCount
#     RETURN graphName, nodeCount, relationshipCount
#     """,
#     params={"nodeQuery": node_query, "relQuery": relationship_query},
# )
# print("Projection result:", projection_result)


# G = "entities"  # Für alle weiteren GDS-Aufrufe reicht der Graph-Name

# # ============================================================
# # 3) k-NN-Graph + WCC auf dem Projection-Graph
# # ============================================================

# similarity_threshold = 0.95

# gds.knn.mutate(
#     G,
#     nodeProperties=["embedding"],
#     mutateRelationshipType="SIMILAR",
#     mutateProperty="score",
#     similarityCutoff=similarity_threshold,
#     topK=10,
# )

# # Weakly Connected Components
# gds.wcc.write(
#     G,
#     writeProperty="wcc",
#     relationshipTypes=["SIMILAR"],
# )

# # ============================================================
# # 4) Kandidaten-Gruppen per Edit-Distance-Filter
# # ============================================================

# word_edit_distance = 3
# potential_duplicate_candidates = graph.query(
#     """
#     MATCH (e:__Entity__)
#     WHERE size(e.id) > 3        // längere Namen
#     WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
#     WHERE count > 1
#     UNWIND nodes AS node
#     // Edit-Distance-Filter
#     WITH DISTINCT
#       [n IN nodes
#        WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance
#           OR node.id CONTAINS n.id | n.id] AS intermediate_results
#     WHERE size(intermediate_results) > 1
#     WITH collect(intermediate_results) AS results
#     // Gruppen verschmelzen, wenn sie Elemente teilen
#     UNWIND range(0, size(results)-1, 1) as index
#     WITH results, index, results[index] as result
#     WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
#             CASE WHEN index <> index2 AND
#                 size(apoc.coll.intersection(acc, results[index2])) > 0
#                 THEN apoc.coll.union(acc, results[index2])
#                 ELSE acc
#             END
#     )) as combinedResult
#     WITH DISTINCT combinedResult as combinedResult
#     WITH collect(combinedResult) as allCombinedResults
#     UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
#     WITH allCombinedResults[combinedResultIndex] as combinedResult,
#          combinedResultIndex, allCombinedResults
#     WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
#         WHERE x <> combinedResultIndex
#           AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
#     )
#     RETURN combinedResult
#     """,
#     params={"distance": word_edit_distance},
# )

# for row in potential_duplicate_candidates:
#     print("Candidate group:", row["combinedResult"])

# # ============================================================
# # 5) LLM-basierte Entscheidung, welche Kandidaten wirklich gemerged werden
# # ============================================================

# system_prompt = """You are an expert in Arduino electronics and embedded hardware. 
# You are given a list of __Entity__ names and descriptions extracted from technical documentation.
# Your task is to detect duplicate entities and decide which ones represent the same real-world hardware component.

# Rules:
# 1. Merge names that describe the same physical component, connector, chip, module, electrical pin, or power source.
# 2. Ignore formatting differences, typos, commas, and order differences.
# 3. If the descriptions refer to the same voltage, connector type, or mechanical port, merge them.
# 4. Do NOT merge different components or variants.
# 5. Prefer the most specific and technically complete name when merging.
# """

# user_template = """
# Here is the list of entities to process:
# {entities}

# Please identify duplicates, merge them, and provide the merged list.
# """

# class DuplicateEntities(BaseModel):
#     entities: List[str] = Field(
#         description="Entities that represent the same object or real-world __Entity__ and should be merged"
#     )

# class Disambiguate(BaseModel):
#     merge_entities: Optional[List[DuplicateEntities]] = Field(
#         description="Lists of entities that represent the same object or real-world __Entity__ and should be merged"
#     )

# extraction_llm = ChatOpenAI(model_name="gpt-4o").with_structured_output(Disambiguate)

# extraction_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", user_template),
#     ]
# )

# extraction_chain = extraction_prompt | extraction_llm

# def entity_resolution(entities: List[str]) -> Optional[List[List[str]]]:
#     result = extraction_chain.invoke({"entities": entities})
#     if not result.merge_entities:
#         return None
#     return [el.entities for el in result.merge_entities]

# merged_entities: List[List[str]] = []

# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     futures = [
#         executor.submit(entity_resolution, el["combinedResult"])
#         for el in potential_duplicate_candidates
#     ]

#     for future in tqdm(as_completed(futures),
#                        total=len(futures),
#                        desc="Processing entity groups"):
#         to_merge = future.result()
#         if to_merge:
#             merged_entities.extend(to_merge)

# print(f"Number of merge groups: {len(merged_entities)}")
# print("Merged entities example:", merged_entities[:5])

# # ============================================================
# # 6) Merge in Neo4j mit APOC
# # ============================================================

# merge_result = graph.query(
#     """
#     UNWIND $data AS candidates
#     CALL {
#       WITH candidates
#       MATCH (e:__Entity__) WHERE e.id IN candidates
#       RETURN collect(e) AS nodes
#     }
#     CALL apoc.refactor.mergeNodes(nodes, {
#       properties: {
#         description: 'combine',
#         `.*`: 'discard'
#       }
#     })
#     YIELD node
#     RETURN count(*) AS mergedCount
#     """,
#     params={"data": merged_entities},
# )

# print("Merge result from Neo4j:", merge_result)
