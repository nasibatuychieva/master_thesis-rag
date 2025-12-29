import json
from typing import Any, Dict, List
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# Hardcoded config (no env)
# =========================
load_dotenv()

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmakg"

COMM_LABEL = "__Community__"
ENTITY_LABEL = "entity"
CHUNK_LABEL = "Chunk"

LEVEL_TO_SUMMARIZE = 1

# how much content per community
TOP_ENTITIES = 30
TOP_PRODUCTS = 15
TOP_CATEGORIES = 15
TOP_CHUNKS = 12

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# -------------------------
# Cypher: build community content
# -------------------------
COMMUNITY_CONTENT_QUERY = f"""
MATCH (c:{COMM_LABEL} {{level: $level}})
OPTIONAL MATCH (e:{ENTITY_LABEL})-[:IN_COMMUNITY]->(c)
WITH c, collect(DISTINCT e) AS ents

WITH c,
     [x IN ents WHERE x.id IS NOT NULL | x.id][0..$top_entities] AS entity_names,
     [x IN ents WHERE x.product IS NOT NULL | x.product] AS all_products,
     [x IN ents WHERE x.product_category IS NOT NULL | x.product_category] AS all_categories

WITH c,
     entity_names,
     apoc.coll.frequencies(all_products) AS prod_freqs,
     apoc.coll.frequencies(all_categories) AS cat_freqs

WITH c,
     entity_names,
     [p IN prod_freqs | {{value: p.item, count: p.count}}] AS prods_raw,
     [p IN cat_freqs | {{value: p.item, count: p.count}}] AS cats_raw

WITH c,
     entity_names,
     apoc.coll.sortMaps(prods_raw, "count")[-$top_products..] AS top_products_desc,
     apoc.coll.sortMaps(cats_raw, "count")[-$top_categories..] AS top_categories_desc

OPTIONAL MATCH (ch:{CHUNK_LABEL})-[:MENTIONS]->(e2:{ENTITY_LABEL})-[:IN_COMMUNITY]->(c)
WITH c, entity_names, top_products_desc, top_categories_desc,
     collect(DISTINCT {{
        chunk_id: ch.chunk_id,
        text: ch.text,
        file: ch.file,
        product: ch.product,
        product_category: ch.product_category
     }}) AS chunks

WITH c, entity_names, top_products_desc, top_categories_desc,
     [x IN chunks WHERE x.text IS NOT NULL][0..$top_chunks] AS sample_chunks

RETURN
  c.communityId AS communityId,
  c.level AS level,
  entity_names AS entities,
  top_products_desc AS top_products,
  top_categories_desc AS top_categories,
  sample_chunks AS chunks
ORDER BY c.communityId
"""


# -------------------------
# LLM prompt
# -------------------------
SYSTEM_MSG = (
    "You are a technical documentation analyst. "
    "Write concise, factual community summaries. No preamble. No bullet spam. "
    "Prefer concrete terms (products, components, actions) and avoid vague wording."
)

HUMAN_TEMPLATE = """You are given a graph community extracted from technical documentation.

Community metadata:
- level: {level}
- communityId: {communityId}

Top entities (names):
{entities}

Top products (value,count):
{top_products}

Top product categories (value,count):
{top_categories}

Representative text snippets (chunk_id, product, category, file, text):
{chunks}

Task:
1) Write a compact "topic label" (max 10 words).
2) Write a 5-10 sentence summary describing what this community is mainly about.
3) List 5-10 key terms.

Output STRICT JSON with keys:
topic_label, summary, key_terms
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("human", HUMAN_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
chain = prompt | llm | StrOutputParser()


def to_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_full_content(row: Dict[str, Any]) -> str:
    # full_content is stored and used for retrieval
    content = {
        "level": row["level"],
        "communityId": row["communityId"],
        "entities": row.get("entities", []),
        "top_products": row.get("top_products", []),
        "top_categories": row.get("top_categories", []),
        "chunks": row.get("chunks", []),
    }
    return to_pretty(content)


def safe_parse_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # fallback: store raw
        return {"topic_label": "", "summary": s, "key_terms": []}


def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        print("\n=== Script 4: Build community full_content + LLM summary ===\n")

        print(f"[1] Fetching communities at level={LEVEL_TO_SUMMARIZE} ...")
        rows = session.run(
            COMMUNITY_CONTENT_QUERY,
            level=LEVEL_TO_SUMMARIZE,
            top_entities=TOP_ENTITIES,
            top_products=TOP_PRODUCTS,
            top_categories=TOP_CATEGORIES,
            top_chunks=TOP_CHUNKS,
        ).data()

        print(f"[INFO] communities fetched: {len(rows)}")

        updates: List[Dict[str, Any]] = []

        print("[2] Generating summaries with LLM ...")
        for i, r in enumerate(rows, start=1):
            full_content = build_full_content(r)

            llm_input = {
                "level": r["level"],
                "communityId": r["communityId"],
                "entities": to_pretty(r.get("entities", [])),
                "top_products": to_pretty(r.get("top_products", [])),
                "top_categories": to_pretty(r.get("top_categories", [])),
                "chunks": to_pretty(r.get("chunks", [])),
            }
            out = chain.invoke(llm_input)
            parsed = safe_parse_json(out)

            updates.append({
                "level": r["level"],
                "communityId": r["communityId"],
                "full_content": full_content,
                "topic_label": parsed.get("topic_label", ""),
                "summary": parsed.get("summary", ""),
                "key_terms": parsed.get("key_terms", []),
            })

            if i % 10 == 0:
                print(f"  ... summarized {i}/{len(rows)}")

        print("[3] Writing back to Neo4j ...")
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (c:{COMM_LABEL} {{level: row.level, communityId: row.communityId}})
            SET c.full_content = row.full_content,
                c.topic_label = row.topic_label,
                c.summary = row.summary,
                c.key_terms = row.key_terms
            """,
            rows=updates,
        )

        print("\nScript 4 finished successfully.\n")

    driver.close()


if __name__ == "__main__":
    main()
