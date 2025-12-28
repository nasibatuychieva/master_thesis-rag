import os
import json
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# FIXED CONFIG
# -----------------------------
URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "simplekg"

COMM_LABEL = "__Community__"
ENTITY_LABEL = "__Entity__"
CHUNK_LABEL_CONTAINS = "Chunk"

REL_IN_COMM = "IN_COMMUNITY"
REL_FROM_CHUNK = "FROM_CHUNK"

# LLM
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set if you want
client = OpenAI()

# how much text we put into the LLM
MAX_SNIPPETS_PER_COMMUNITY = 60
SNIPPET_CHAR_LIMIT = 350         # truncate each chunk text
FULL_CONTENT_CHAR_LIMIT = 40_000 # cap total to avoid context explosion

# writing
WRITE_FULL_CONTENT = True
WRITE_SUMMARY = True

def _safe_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip()
    # keep it single-line-ish
    s = " ".join(s.split())
    return s

def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."

def fetch_community_ids(session) -> List[str]:
    # Use elementId (Neo4j 5+ recommended)
    rows = session.run(f"""
        MATCH (c:{COMM_LABEL})
        WHERE c.level = 1
        RETURN elementId(c) AS cid, c.communityId AS communityId
        ORDER BY c.communityId
    """).data()
    return [r["cid"] for r in rows]

def fetch_snippets_for_community(session, comm_eid: str, limit: int) -> List[str]:
    # Your chunks have property `text` (based on your diagnostics)
    rows = session.run(f"""
        MATCH (e:{ENTITY_LABEL})-[:{REL_IN_COMM}]->(c:{COMM_LABEL})
        WHERE elementId(c) = $cid

        // connect entity <-> chunk (direction doesn’t matter)
        MATCH (e)-[:{REL_FROM_CHUNK}]-(ch)
        WHERE any(l IN labels(ch) WHERE l CONTAINS $chunkLabelContains)

        WITH DISTINCT ch
        RETURN ch.text AS snippet
        LIMIT $limit
    """, cid=comm_eid, limit=limit, chunkLabelContains=CHUNK_LABEL_CONTAINS).data()

    snippets = []
    for r in rows:
        t = _safe_text(r.get("snippet"))
        if not t:
            continue
        snippets.append(_truncate(t, SNIPPET_CHAR_LIMIT))
    return snippets

def build_full_content(snippets: List[str]) -> str:
    # dedupe while keeping order
    seen = set()
    uniq = []
    for s in snippets:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    full = "\n\n".join(uniq)
    full = full.strip()
    if len(full) > FULL_CONTENT_CHAR_LIMIT:
        full = full[:FULL_CONTENT_CHAR_LIMIT] + "\n\n[TRUNCATED]"
    return full

def summarize_with_llm(full_content: str) -> Dict[str, Any]:
    """
    Returns dict with keys:
      - summary: str
      - keywords: list[str]
    Robust: forces JSON output. Also tolerant fallback parse.
    """
    prompt = (
        "You are summarizing a knowledge-graph community.\n"
        "Given the text snippets below, produce a concise community summary and 5-15 keywords.\n"
        "Return ONLY valid JSON with exactly these keys: summary (string), keywords (array of strings).\n\n"
        "SNIPPETS:\n"
        f"{full_content}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only JSON. No markdown. No extra keys."},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content or "{}"

    # Should already be JSON because of response_format, but still safe:
    try:
        obj = json.loads(content)
        if "summary" not in obj:
            obj["summary"] = ""
        if "keywords" not in obj or not isinstance(obj["keywords"], list):
            obj["keywords"] = []
        obj["summary"] = str(obj["summary"]).strip()
        obj["keywords"] = [str(x).strip() for x in obj["keywords"] if str(x).strip()]
        return obj
    except Exception:
        # absolute fallback: store raw
        return {"summary": "", "keywords": [], "raw": content}

def write_results(session, comm_eid: str, full_content: str, llm_out: Dict[str, Any]):
    updates = []
    if WRITE_FULL_CONTENT:
        updates.append("c.full_content = $full_content")
    if WRITE_SUMMARY:
        updates.append("c.summary = $summary")
        updates.append("c.keywords = $keywords")

    if not updates:
        return

    session.run(f"""
        MATCH (c:{COMM_LABEL})
        WHERE elementId(c) = $cid
        SET {", ".join(updates)}
    """, cid=comm_eid, full_content=full_content,
       summary=llm_out.get("summary", ""),
       keywords=llm_out.get("keywords", []))

def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=DATABASE) as session:
        cids = fetch_community_ids(session)
        print(f"[INFO] Found level=1 communities: {len(cids)}")

        done = 0
        skipped_empty = 0

        for cid in cids:
            snippets = fetch_snippets_for_community(session, cid, MAX_SNIPPETS_PER_COMMUNITY)
            full_content = build_full_content(snippets)

            if not full_content.strip():
                skipped_empty += 1
                continue

            llm_out = summarize_with_llm(full_content)
            write_results(session, cid, full_content, llm_out)

            done += 1
            if done % 10 == 0:
                print(f"[PROGRESS] summarized {done}/{len(cids)} (skipped_empty={skipped_empty})")

        print(f"\n[DONE] summarized={done}, skipped_empty={skipped_empty}, total={len(cids)}")

    driver.close()

if __name__ == "__main__":
    main()
