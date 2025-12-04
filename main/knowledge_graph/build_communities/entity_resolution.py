# https://neo4j.com/blog/developer/global-graphrag-neo4j-langchain/
# Our process for entity resolution involves the following steps:

# Entities in the graph — Start with all entities within the graph.
# K-nearest graph — Construct a k-nearest neighbor graph, connecting similar entities based on text embeddings.
# Weakly Connected Components — Identify weakly connected components in the k-nearest graph, grouping entities that are 
# likely to be similar. 
# Add a word distance filtering step after these components have been identified.
# LLM evaluation — Use an LLM to evaluate these components and decide whether the entities within each component should be 
# merged, resulting in a final decision on entity resolution (for example, merging ‘Silicon Valley Bank’ and ‘
# Silicon_Valley_Bank’ while rejecting the merge for different dates like ‘September 16, 2023’ and ‘September 2, 2023’).
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 10
NUM_ARTICLES = 2000
graph_documents = []

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "testmaster123"

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

#USE_OPENAI_EMBEDDINGS = True  # via .env oder config

#if USE_OPENAI_EMBEDDINGS:
from langchain_openai import OpenAIEmbeddings
embedding_provider = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
EMBED_DIM = 1536
# else:
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     embedding_provider = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     EMBED_DIM = 384

# Wichtig: Label & Properties an deine Struktur anpassen!
vector = Neo4jVector.from_existing_graph(
    embedding=embedding_provider,
    graph=graph,
    node_label="Entity",                 # z.B. :Entity oder :Element etc.
    text_node_properties=["id", "description"],
    embedding_node_property="embedding"  # wird als Property auf den Nodes gespeichert
)

print("Embeddings geschrieben.")

from graphdatascience import GraphDataScience
import os



gds = GraphDataScience(
    URI,
    auth=(AUTH_USER, AUTH_PASSWORD ),
)
# Falls der Graph "entities" schon existiert, zuerst löschen
gds.run_cypher("CALL gds.graph.drop('entities', false)")


# To create the k-nearest neighbor graph, we will project all entities along with their text embeddings:
G, result = gds.graph.project(
    "entities",
    "Entity",               
    "*",                    
    nodeProperties=["embedding"]
)

print(G, result)


# k-nearest graph
similarity_threshold = 0.95  

gds.knn.mutate(
    G,
    nodeProperties=["embedding"],
    mutateRelationshipType="SIMILAR",
    mutateProperty="score",
    
    similarityCutoff=similarity_threshold,
    topK=10,  # k-Nachbarn
)
# Weakly Connected Components algorithm
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)
word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    """
    MATCH (e:Entity)
    WHERE size(e.id) > 3 // längere Namen
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Edit Distance Filter
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                  OR node.id CONTAINS n.id | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // Gruppen verschmelzen, wenn sie Elemente teilen
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
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """,
    params={"distance": word_edit_distance},
)

for row in potential_duplicate_candidates:
    print("Candidate group:", row["combinedResult"])


system_prompt = """You are an expert in Arduino electronics and embedded hardware. 
You are given a list of entity names and descriptions extracted from technical documentation.
Your task is to detect duplicate entities and decide which ones represent the same real-world hardware component.

Rules:
1. Merge names that describe the same physical component, connector, chip, module, electrical pin, or power source.
2. Ignore formatting differences, typos, commas, and order differences (e.g., "X2, Power Jack" ≈ "Power Jack").
3. If the descriptions refer to the same voltage range, electrical connector, or mechanical port, merge them (e.g., "Power Jack Vin 7–12 Vcc" ≈ "DC Barrel Jack").
4. Do NOT merge two different components or variants. For example:
   - USB-C ≠ Micro-USB
   - 3.3V power pin ≠ 5V power pin
   - ATmega2560 processor ≠ SAM3X processor
5. Prefer the most specific and technically complete name when merging. Keep the canonical name explicit (e.g., prefer "Power Jack Vin 7–12 Vcc" over "Power Jack").

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
  MATCH (e:Entity) WHERE e.id IN candidates
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

## dieser Schritt wurde ebenfalls durchgeührt in Neo4j direkt:
##MATCH (e:Entity)
# MATCH (p:Product)
# WHERE toLower(e.id) CONTAINS( toLower(p.name))
# AND e.entityType CONTAINS ("Product")  // Matching-Bedingung
#   AND e <> p
# CALL apoc.refactor.mergeNodes([p, e], {
#   mergeRels: true,      // alle Kanten beider Knoten zusammenführen
#   properties: "discard" // Properties von p behalten, die von e verwerfen
# })
# YIELD node
# RETURN node;
