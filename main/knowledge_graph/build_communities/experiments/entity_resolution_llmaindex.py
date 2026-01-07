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

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
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
The entities might be slightly different in format or content, but essentially refer to the same thing. Use  analytical skills to determine duplicates.

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

