from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph


# -----------------------------------------------------------------------------
# Neo4j Connection
# -----------------------------------------------------------------------------
import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"

MAX_WORKERS = 10
MIN_NORM_LEN = 4  # Schutz gegen zu aggressive Normalisierung bei sehr kurzen Names

# Meta-Labels, die NICHT als "semantische Labels" zählen sollen
META_LABELS = ["__Entity__", "__KGBuilder__"]

graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")


# -----------------------------------------------------------------------------
# 1) Candidate Generation (STRICT numeric-safe + TOKENS) + dedupe
#    -> basiert auf e.name, aber NUR innerhalb gleicher semantischer Label-Signatur
#    FIX: groupKey für normTokens via apoc.text.join(...) statt toString(list)
# -----------------------------------------------------------------------------
candidate_query = f"""
// ============================================================================
// STRICT normalization (numeric-safe) ON name
// + ONLY within same semantic label signature
// - preserves decimals: 2.0 -> 2dot0
// - preserves minus:    -2  -> neg2
// ============================================================================
WITH {MIN_NORM_LEN} AS minLen, {META_LABELS} AS metaLabels
MATCH (e:__Entity__)
WHERE e.name IS NOT NULL

WITH e, toLower(e.name) AS s, minLen,
     apoc.coll.sort([l IN labels(e) WHERE NOT l IN metaLabels]) AS semLabels

WITH e, semLabels,
     apoc.text.replace(s, '(\\d)\\.(\\d)', '$1dot$2') AS s1,
     minLen
WITH e, semLabels,
     apoc.text.replace(s1, '-', 'neg') AS s2,
     minLen
WITH e, semLabels,
     apoc.text.replace(s2, '[^a-z0-9]+', '') AS strictNorm,
     minLen
WHERE strictNorm <> '' AND size(strictNorm) >= minLen

WITH strictNorm AS groupKey, semLabels, collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'STRICT_NUM_SAFE' AS method,
  groupKey,
  semLabels,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.name] AS combinedNames,
  [n IN nodes | elementId(n)] AS elementIds

UNION

// ============================================================================
// TOKEN-based candidates ON name
// + ONLY within same semantic label signature
// ============================================================================
WITH [
  'arduino', 'board', 'boards', 'processor', 'processors',
  'sensor', 'sensors', 'module', 'modules', 'shield', 'shields',
  'kit', 'kits', 'dev', 'development'
] AS genericWords,
{MIN_NORM_LEN} AS minLen,
{META_LABELS} AS metaLabels

MATCH (e:__Entity__)
WHERE e.name IS NOT NULL

WITH e, genericWords, minLen,
     apoc.coll.sort([l IN labels(e) WHERE NOT l IN metaLabels]) AS semLabels,
     apoc.text.replace(toLower(e.name), '[^a-z0-9]+', ' ') AS cleaned

WITH e, semLabels, minLen,
     [w IN split(cleaned, ' ')
       WHERE w <> '' AND NOT w IN genericWords] AS tokens

WITH e, semLabels, minLen,
     apoc.coll.sort(apoc.coll.toSet(tokens)) AS normTokens

WHERE size(normTokens) > 0
  AND any(t IN normTokens WHERE size(t) >= minLen)

// FIX: stable string key from list of tokens
WITH apoc.text.join(normTokens, '|') AS groupKey, semLabels, collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'TOKENS' AS method,
  groupKey,
  semLabels,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.name] AS combinedNames,
  [n IN nodes | elementId(n)] AS elementIds
"""

potential_duplicate_candidates = graph.query(candidate_query)
print("Duplicate candidate groups (raw):", len(potential_duplicate_candidates))


def dedupe_candidate_groups(rows):
    """
    Dedupe overlapping groups by node identity (elementIds).
    (STRICT and TOKENS can sometimes produce the same node set.)
    """
    seen = set()
    out = []
    for r in rows:
        eids = r.get("elementIds") or []
        if len(eids) < 2:
            continue
        key = tuple(sorted(eids))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


potential_duplicate_candidates = dedupe_candidate_groups(potential_duplicate_candidates)
print("Duplicate candidate groups (deduped):", len(potential_duplicate_candidates))


# -----------------------------------------------------------------------------
# 2) LLM-based grouping (structured output)
#    -> LLM bekommt NUR Namen
# -----------------------------------------------------------------------------
system_prompt = """
You are a data processing assistant. Your task is to identify duplicate entities based on their NAMES.

Rules:
- Only group items that refer to the same real-world entity.
- Treat differences in punctuation, parentheses, hyphens, underscores, and whitespace as insignificant.
- IMPORTANT: Do NOT merge values with different numeric meaning (e.g., 20 vs 2.0, -2 vs -2.0).
- If two names differ only by non-alphanumeric characters (and numeric meaning is unchanged), they can be grouped.
Return groups to merge. If nothing should be merged, return null/empty.
"""

user_template = """
Here is the list of entity names to process:
{entities}

Identify duplicates and group them together (groups to merge).
"""

class DuplicateEntities(BaseModel):
    entities: List[str]

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]]

extraction_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Disambiguate)

extraction_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", user_template)]
)

extraction_chain = extraction_prompt | extraction_llm


def entity_resolution(names: List[str]) -> Optional[List[List[str]]]:
    result = extraction_chain.invoke({"entities": names})
    if not result.merge_entities:
        return None
    return [el.entities for el in result.merge_entities]


# -----------------------------------------------------------------------------
# 3) Run LLM in parallel
# -----------------------------------------------------------------------------
merged_name_groups: List[List[str]] = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(entity_resolution, row["combinedNames"])
        for row in potential_duplicate_candidates
        if row.get("combinedNames") and len(row.get("combinedNames")) > 1
    ]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        res = future.result()
        if res:
            merged_name_groups.extend(res)

print("Duplicate merge groups (LLM output):", len(merged_name_groups))


# -----------------------------------------------------------------------------
# 4) Resolve merged NAME groups -> elementIds
#    IMPORTANT: name may not be unique => we collect all matching nodes.
# -----------------------------------------------------------------------------
resolve_to_element_ids_query = """
UNWIND $names AS name
MATCH (e:__Entity__)
WHERE e.name = name
RETURN name AS name, collect(elementId(e)) AS eids
"""

def group_names_to_element_ids(group: List[str]) -> List[str]:
    rows = graph.query(resolve_to_element_ids_query, params={"names": group})
    eids: List[str] = []
    for r in rows:
        eids.extend(r.get("eids", []))
    return list(dict.fromkeys(eids))


merge_groups_element_ids: List[List[str]] = []
for g in merged_name_groups:
    eids = group_names_to_element_ids(g)
    if len(eids) > 1:
        merge_groups_element_ids.append(eids)

# Dedupe merge groups (same node sets can appear multiple times)
merge_groups_element_ids = [list(x) for x in {tuple(sorted(g)) for g in merge_groups_element_ids}]
print("Duplicate merge groups (elementId-based):", len(merge_groups_element_ids))


# -----------------------------------------------------------------------------
# 5) Merge in Neo4j (always merge by elementId)
# -----------------------------------------------------------------------------
merge_query = """
UNWIND $data AS eids
CALL {
  WITH eids
  MATCH (e:__Entity__)
  WHERE elementId(e) IN eids
  RETURN collect(e) AS nodes
}
WITH nodes
WHERE size(nodes) > 1
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
"""

result = graph.query(merge_query, params={"data": merge_groups_element_ids})
print("Merge result from Neo4j:", result)

print("\nSample merged groups (up to 10):")
for i, g in enumerate(merge_groups_element_ids[:10], start=1):
    print(f"{i:02d} -> {g}")
