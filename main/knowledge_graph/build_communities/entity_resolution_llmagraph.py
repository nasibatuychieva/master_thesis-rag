from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"  # ggf. korrigieren

MAX_WORKERS = 10

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")

potential_duplicate_candidates = graph.query(
    """
      WITH [
        'arduino', 'board', 'boards', 'processor', 'processors',
        'sensor', 'sensors', 'module', 'modules', 'shield', 'shields',
        'kit', 'kits', 'dev', 'development'
      ] AS genericWords

      MATCH (e:Entity)
      WHERE e.id IS NOT NULL
        AND e.entityType IS NOT NULL

      WITH e, genericWords,
          apoc.text.replace(toLower(e.id), '[^a-z0-9]+', ' ') AS cleaned

      WITH e,
          [w IN split(cleaned, ' ')
            WHERE w <> '' AND NOT w IN genericWords] AS tokens

      WITH e,
          apoc.coll.sort(apoc.coll.toSet(tokens)) AS normTokens

      WHERE size(normTokens) > 0

      WITH e.entityType AS entityType, normTokens, collect(e) AS nodes
      WHERE size(nodes) > 1

      RETURN
        entityType,
        normTokens,
        size(nodes) AS duplicateCount,
        [n IN nodes | n.id] AS combinedResult
      ORDER BY duplicateCount DESC, entityType, normTokens;
    """
)

print("Duplicate candidate groups:", len(potential_duplicate_candidates))

system_prompt = """
You are a data processing assistant. Your task is to identify duplicate entities.
"""

user_template = """
Here is the list of entities to process:
{entities}

Identify duplicates and group them together.
"""

class DuplicateEntities(BaseModel):
    entities: List[str]

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]]

extraction_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Disambiguate)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", user_template),
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
    futures = [
        executor.submit(entity_resolution, el["combinedResult"])
        for el in potential_duplicate_candidates
    ]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        res = future.result()
        if res:
            merged_entities.extend(res)

print("Duplicate merge groups:", len(merged_entities))

result = graph.query(
    """
    UNWIND $data AS candidates
    CALL {
      WITH candidates
      MATCH (e:Entity) 
      WHERE e.id IN candidates
      RETURN collect(e) AS nodes
    }
    CALL apoc.refactor.mergeNodes(
      nodes,
      {
        mergeRels: true,
        properties: {
          description: 'combine',
          `.*`: 'discard'
        }
      }
    )
    YIELD node
    RETURN count(*) AS mergedGroups
    """,
    params={"data": merged_entities}
)

print("Merge result from Neo4j:", result)
