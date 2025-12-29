from dotenv import load_dotenv
load_dotenv()

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict

from tqdm import tqdm
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph

from openai import RateLimitError
from neo4j.exceptions import ClientError


# =============================================================================
# CONFIG
# =============================================================================
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmagraphtrkg"


MAX_WORKERS = 2

MIN_NORM_LEN = 4


MAX_ENTITIES_PER_GROUP = 80

# LLM retry/backoff
MAX_RETRIES = 8




# =============================================================================
# CONNECT
# =============================================================================
graph = Neo4jGraph(
    url=URI,
    username=AUTH_USER,
    password=AUTH_PASSWORD,
    refresh_schema=False,
    database=DATABASE,
)

print("\n=== Connected to Neo4j Knowledge Graph ===\n")


# =============================================================================
# 1) CANDIDATE GENERATION (STRICT + STRICT_PLURAL_SAFE + TOKENS + TOKENS_PLURAL_SAFE)
# =============================================================================
candidate_query = f"""
WITH {MIN_NORM_LEN} AS minLen

// ============================================================================
// STRICT numeric-safe (original)
// ============================================================================
MATCH (e:Entity)
WHERE e.id IS NOT NULL AND e.entityType IS NOT NULL

WITH e, toLower(e.id) AS s, minLen
WITH e, apoc.text.replace(s, '(\\d)\\.(\\d)', '$1dot$2') AS s1, minLen
WITH e, apoc.text.replace(s1, '-', 'neg') AS s2, minLen
WITH e, apoc.text.replace(s2, '[^a-z0-9]+', '') AS strictNorm, minLen
WHERE strictNorm <> '' AND size(strictNorm) >= minLen

WITH e.entityType AS entityType, strictNorm AS groupKey, collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'STRICT_NUM_SAFE' AS method,
  entityType,
  groupKey,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.id] AS combinedResult,
  [n IN nodes | elementId(n)] AS combinedElementIds

UNION

// ============================================================================
// STRICT numeric-safe + PLURAL-SAFE
// - remove ONE trailing 's' if the strictNorm ends with 's' and is long enough
// - heuristic: avoids touching short strings; keeps numeric meaning intact
// ============================================================================
WITH {MIN_NORM_LEN} AS minLen
MATCH (e:Entity)
WHERE e.id IS NOT NULL AND e.entityType IS NOT NULL

WITH e, toLower(e.id) AS s, minLen
WITH e, apoc.text.replace(s, '(\\d)\\.(\\d)', '$1dot$2') AS s1, minLen
WITH e, apoc.text.replace(s1, '-', 'neg') AS s2, minLen
WITH e, apoc.text.replace(s2, '[^a-z0-9]+', '') AS strictNorm, minLen
WHERE strictNorm <> '' AND size(strictNorm) >= minLen

WITH e, strictNorm,
     CASE
       WHEN size(strictNorm) >= (minLen + 2) AND strictNorm ENDS WITH 's'
            THEN substring(strictNorm, 0, size(strictNorm) - 1)
       ELSE strictNorm
     END AS strictNormPluralSafe

WITH e.entityType AS entityType, strictNormPluralSafe AS groupKey, collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'STRICT_NUM_SAFE_PLURAL_SAFE' AS method,
  entityType,
  groupKey,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.id] AS combinedResult,
  [n IN nodes | elementId(n)] AS combinedElementIds

UNION

// ============================================================================
// TOKENS (original) with stable join
// ============================================================================
WITH [
  'arduino', 'board', 'boards', 'processor', 'processors',
  'sensor', 'sensors', 'module', 'modules', 'shield', 'shields',
  'kit', 'kits', 'dev', 'development'
] AS genericWords,
{MIN_NORM_LEN} AS minLen

MATCH (e:Entity)
WHERE e.id IS NOT NULL AND e.entityType IS NOT NULL

WITH e, genericWords, minLen,
     apoc.text.replace(toLower(e.id), '[^a-z0-9]+', ' ') AS cleaned

WITH e, minLen,
     [w IN split(cleaned, ' ')
       WHERE w <> '' AND NOT w IN genericWords] AS tokens

WITH e, minLen,
     apoc.coll.sort(apoc.coll.toSet(tokens)) AS normTokens

WHERE size(normTokens) > 0
  AND any(t IN normTokens WHERE size(t) >= minLen)

WITH e.entityType AS entityType,
     apoc.text.join(normTokens, '|') AS groupKey,
     collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'TOKENS' AS method,
  entityType,
  groupKey,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.id] AS combinedResult,
  [n IN nodes | elementId(n)] AS combinedElementIds

UNION

// ============================================================================
// TOKENS + PLURAL-SAFE
// - singularize tokens by stripping ONE trailing 's' for tokens length >= 4
// - keeps 'ss' unchanged and avoids very short tokens
// ============================================================================
WITH [
  'arduino', 'board', 'boards', 'processor', 'processors',
  'sensor', 'sensors', 'module', 'modules', 'shield', 'shields',
  'kit', 'kits', 'dev', 'development'
] AS genericWords,
{MIN_NORM_LEN} AS minLen

MATCH (e:Entity)
WHERE e.id IS NOT NULL AND e.entityType IS NOT NULL

WITH e, genericWords, minLen,
     apoc.text.replace(toLower(e.id), '[^a-z0-9]+', ' ') AS cleaned

WITH e, minLen,
     [w IN split(cleaned, ' ')
       WHERE w <> '' AND NOT w IN genericWords] AS tokens

WITH e, minLen,
     [t IN tokens |
        CASE
          WHEN size(t) >= 4 AND t ENDS WITH 's' AND NOT t ENDS WITH 'ss'
               THEN substring(t, 0, size(t) - 1)
          ELSE t
        END
     ] AS tokensPluralSafe

WITH e, minLen,
     apoc.coll.sort(apoc.coll.toSet(tokensPluralSafe)) AS normTokensPluralSafe

WHERE size(normTokensPluralSafe) > 0
  AND any(t IN normTokensPluralSafe WHERE size(t) >= minLen)

WITH e.entityType AS entityType,
     apoc.text.join(normTokensPluralSafe, '|') AS groupKey,
     collect(e) AS nodes
WHERE size(nodes) > 1

RETURN
  'TOKENS_PLURAL_SAFE' AS method,
  entityType,
  groupKey,
  size(nodes) AS duplicateCount,
  [n IN nodes | n.id] AS combinedResult,
  [n IN nodes | elementId(n)] AS combinedElementIds
"""

potential_duplicate_candidates = graph.query(candidate_query)
print("Duplicate candidate groups (raw):", len(potential_duplicate_candidates))


def dedupe_candidate_groups(rows: List[Dict]) -> List[Dict]:
    """Dedupe identical candidate groups by node identity (elementIds)."""
    seen = set()
    out = []
    for r in rows:
        eids = r.get("combinedElementIds") or []
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


# =============================================================================
# 2) LLM GROUPING
# =============================================================================
system_prompt = """
You are a data processing assistant. Your task is to identify duplicate entities.

Rules:
- Only group items that refer to the same real-world entity.
- Treat differences in punctuation, parentheses, hyphens, underscores, whitespace, and pluralization (e.g., Expansion vs Expansions)
  as insignificant IF the meaning is the same.
- IMPORTANT: Do NOT merge values with different numeric meaning (e.g., 20 vs 2.0, -2 vs -2.0).
- If two IDs differ only by non-alphanumeric characters (and numeric meaning is unchanged), they can be grouped.
Return groups to merge. If nothing should be merged, return null/empty.
"""

user_template = """
Here is the list of entity IDs to process (same entityType / same candidate group):
{entities}

Identify duplicates and group them together (groups to merge).
"""

class DuplicateEntities(BaseModel):
    entities: List[str]

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]]

extraction_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Disambiguate)
extraction_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_template)])
extraction_chain = extraction_prompt | extraction_llm


def entity_resolution_with_retry(entities: List[str], max_retries: int = MAX_RETRIES) -> Optional[List[List[str]]]:
    if len(entities) > MAX_ENTITIES_PER_GROUP:
        entities = entities[:MAX_ENTITIES_PER_GROUP]

    for attempt in range(max_retries):
        try:
            result = extraction_chain.invoke({"entities": entities})
            if not result.merge_entities:
                return None
            return [el.entities for el in result.merge_entities]

        except RateLimitError:
            sleep_s = (2 ** attempt) * 0.5 + random.random() * 0.25
            time.sleep(sleep_s)
        except Exception:
            sleep_s = (2 ** attempt) * 0.25 + random.random() * 0.25
            time.sleep(sleep_s)

    return None


# =============================================================================
# 3) RUN LLM (LOW CONCURRENCY)
# =============================================================================
merged_entities: List[List[str]] = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(entity_resolution_with_retry, row["combinedResult"])
        for row in potential_duplicate_candidates
        if row.get("combinedResult") and len(row["combinedResult"]) > 1
    ]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        res = future.result()
        if res:
            merged_entities.extend(res)

print("Duplicate merge groups (LLM output):", len(merged_entities))


# =============================================================================
# 4) RESOLVE LLM GROUPS (IDs) -> elementIds
# =============================================================================
resolve_to_element_ids_query = """
UNWIND $ids AS id
MATCH (e:Entity)
WHERE e.id = id
RETURN id AS id, collect(elementId(e)) AS eids
"""

def group_ids_to_element_ids(group: List[str]) -> List[str]:
    rows = graph.query(resolve_to_element_ids_query, params={"ids": group})
    eids: List[str] = []
    for r in rows:
        eids.extend(r.get("eids", []))
    return list(dict.fromkeys(eids))


merge_groups_element_ids: List[List[str]] = []
for g in merged_entities:
    eids = group_ids_to_element_ids(g)
    if len(eids) > 1:
        merge_groups_element_ids.append(eids)

# Deduplicate identical groups
merge_groups_element_ids = [list(x) for x in {tuple(sorted(g)) for g in merge_groups_element_ids}]
print("Duplicate merge groups (elementId-based):", len(merge_groups_element_ids))


# =============================================================================
# 4b) MERGE OVERLAPPING GROUPS (UNION-FIND) to avoid NotFound during merge
# =============================================================================
class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


uf = UnionFind()
for group in merge_groups_element_ids:
    root = group[0]
    for x in group[1:]:
        uf.union(root, x)

components: Dict[str, List[str]] = {}
for group in merge_groups_element_ids:
    for x in group:
        r = uf.find(x)
        components.setdefault(r, [])
        components[r].append(x)

merged_components: List[List[str]] = []
seen_comp = set()
for _, xs in components.items():
    uniq = list(dict.fromkeys(xs))
    if len(uniq) > 1:
        key = tuple(sorted(uniq))
        if key not in seen_comp:
            seen_comp.add(key)
            merged_components.append(uniq)

merge_groups_element_ids = merged_components
print("Merge groups after overlap-union (elementId-based):", len(merge_groups_element_ids))


# =============================================================================
# 5) MERGE IN NEO4J (ROBUST: one-by-one)
# =============================================================================
merge_one_group_query = """
MATCH (e:Entity)
WHERE elementId(e) IN $eids
WITH collect(DISTINCT e) AS nodes
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
RETURN 1 AS merged
"""

merged_ok = 0
skipped = 0
failed = 0

print("\nMerging groups one-by-one (robust)...")
for eids in tqdm(merge_groups_element_ids, desc="Merging", total=len(merge_groups_element_ids)):
    try:
        res = graph.query(merge_one_group_query, params={"eids": eids})
        if res and res[0].get("merged") == 1:
            merged_ok += 1
        else:
            skipped += 1
    except ClientError:
        failed += 1
        continue

print(f"\nMerge summary: merged_ok={merged_ok}, skipped={skipped}, failed={failed}")


# =============================================================================
# 6) POST-CHECK: quick sample + remaining exact duplicates by id
# =============================================================================
print("\nSample merged groups (up to 10):")
for i, g in enumerate(merge_groups_element_ids[:10], start=1):
    print(f"{i:02d} -> {g}")

remaining_dupes = graph.query("""
MATCH (e:Entity)
WHERE e.id IS NOT NULL
WITH e.id AS id, count(*) AS c
WHERE c > 1
RETURN id, c
ORDER BY c DESC
LIMIT 25
""")
print("\nRemaining exact duplicate IDs (top 25):")
for r in remaining_dupes:
    print(r)
