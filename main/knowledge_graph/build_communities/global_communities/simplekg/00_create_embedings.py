import os

from typing import Any, Dict, List

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()
import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"

ENTITY_LABEL = "__Entity__"
EMBED_PROP = "embedding"

EMBED_MODEL = "text-embedding-3-small"

# batching
READ_BATCH_SIZE = 200
EMBED_BATCH_SIZE = 100  
MAX_TEXT_CHARS = 4000  

# properties to exclude from embedding text
EXCLUDE_PROPS = {"__tmp_internal_id", "<id>", EMBED_PROP}


def _normalize_value(v: Any) -> str:
    """Convert property value to a compact string suitable for embedding text."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        return s
    if isinstance(v, list):
     
        parts = []
        for x in v[:20]:
            xs = _normalize_value(x)
            if xs:
                parts.append(xs)
        s = ", ".join(parts)
        if len(v) > 20:
            s += ", ..."
        return s
    if isinstance(v, dict):
        # compact dict
        items = []
        for k in list(v.keys())[:20]:
            vs = _normalize_value(v[k])
            if vs:
                items.append(f"{k}={vs}")
        s = "; ".join(items)
        if len(v) > 20:
            s += "; ..."
        return s
    return str(v).strip()


def build_embedding_text(labels: List[str], props: Dict[str, Any]) -> str:
    """
    Build a robust text representation:
    - labels (excluding the base label)
    - category, name if present
    - remaining properties as key=value
    Excludes __tmp_internal_id, embedding
    """
    lbls = [l for l in labels if l and l != ENTITY_LABEL]
    cat = _normalize_value(props.get("category"))
    name = _normalize_value(props.get("name"))

    # remaining props
    kv_parts = []
    for k in sorted(props.keys()):
        if k in EXCLUDE_PROPS:
            continue
        if k in ("category", "name"):
            continue
        vs = _normalize_value(props.get(k))
        if vs:
            kv_parts.append(f"{k}={vs}")

    text = (
        f"labels: {', '.join(lbls) if lbls else 'None'}\n"
        f"category: {cat if cat else 'None'}\n"
        f"name: {name if name else 'None'}\n"
        f"properties: {'; '.join(kv_parts) if kv_parts else 'None'}"
    )

    # cap
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    return text


def main():
  
 

    client = OpenAI()

    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    print("\n=== STEP 1: Write embeddings on __Entity__ ===\n")

    with driver.session(database=DATABASE) as session:
        # how many entities total / already embedded
        stat = session.run(f"""
        MATCH (e:{ENTITY_LABEL})
        RETURN
          count(e) AS total,
          count(CASE WHEN e.{EMBED_PROP} IS NOT NULL THEN 1 END) AS with_embedding
        """).single()
        print(f"[Stats] total={stat['total']}, with_embedding={stat['with_embedding']}")

        # We iterate by internal id for stable paging

        last_id = -1
        processed = 0
        written = 0

        while True:
            rows = session.run(f"""
            MATCH (e:{ENTITY_LABEL})
            WHERE id(e) > $last_id
            RETURN id(e) AS nid, labels(e) AS labels, properties(e) AS props
            ORDER BY id(e)
            LIMIT $limit
            """, last_id=last_id, limit=READ_BATCH_SIZE).data()

            if not rows:
                break

            last_id = rows[-1]["nid"]

    
            node_ids: List[int] = []
            texts: List[str] = []

            for r in rows:
                nid = r["nid"]
                labels = r["labels"] or []
                props = r["props"] or {}

           
                text = build_embedding_text(labels, props)

              
                if not text.strip():
                    continue

                node_ids.append(nid)
                texts.append(text)

            processed += len(rows)

            if not texts:
                continue

            # embed in sub-batches
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                sub_texts = texts[i:i + EMBED_BATCH_SIZE]
                sub_ids = node_ids[i:i + EMBED_BATCH_SIZE]

                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=sub_texts
                )

                vectors = [d.embedding for d in resp.data]
                if len(vectors) != len(sub_ids):
                    raise RuntimeError("Embedding response size mismatch.")

                # write back with UNWIND for speed
                session.run(f"""
                UNWIND $rows AS row
                MATCH (e:{ENTITY_LABEL})
                WHERE id(e) = row.nid
                SET e.{EMBED_PROP} = row.vec
                """, rows=[{"nid": nid, "vec": vec} for nid, vec in zip(sub_ids, vectors)])

                written += len(sub_ids)

            print(f"[Progress] processed={processed}/{stat['total']} | wrote_embeddings={written}")

    driver.close()
    print("\Finished.\n")


if __name__ == "__main__":
    main()
