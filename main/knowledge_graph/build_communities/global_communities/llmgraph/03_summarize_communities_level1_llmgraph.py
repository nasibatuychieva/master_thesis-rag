import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------
# FIXED CONFIG (wie gewünscht)
# ---------------------------------
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmagraphtrkg"

MODEL = "gpt-4o-mini"
LEVEL = 1

MAX_WORKERS = 8
MAX_ENTITIES = 80
MAX_CHUNKS = 12

def j(x) -> str:
    return json.dumps(x, ensure_ascii=False)

FETCH_COMMUNITIES = """
MATCH (c:__Community__)
WHERE c.level = $level
RETURN c.communityId AS cid
ORDER BY cid
"""

FETCH_CONTEXT = """
MATCH (c:__Community__ {level:$level, communityId:$cid})
OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
WITH c, collect(DISTINCT e)[0..$maxEntities] AS ents
OPTIONAL MATCH (ch:Chunk)-[:MENTIONS]->(e2:Entity)-[:IN_COMMUNITY]->(c)
WITH c, ents, collect(DISTINCT ch.text)[0..$maxChunks] AS chunks
RETURN
  c.communityId AS cid,
  [e IN ents | {id:e.id, entityType:e.entityType, description:coalesce(e.description,"")}] AS entities,
  chunks AS chunks
"""

UPDATE = """
MATCH (c:__Community__ {level:$level, communityId:$cid})
SET c.full_content = $txt
"""

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You create a compact, factual community summary for retrieval. "
     "Do not invent facts. Use ONLY the provided entities and snippets."),
    ("user",
     "Community level: {level}\nCommunityId: {cid}\n\n"
     "Entities:\n{entities}\n\n"
     "Representative chunk snippets:\n{chunks}\n\n"
     "Write:\n"
     "1) 2-4 sentences overview\n"
     "2) 8-15 bullet points of key facts\n"
     "3) Keywords (comma-separated)\n")
])

def summarize_one(driver, llm, cid: int) -> Dict[str, Any]:
    with driver.session(database=DATABASE) as session:
        data = session.run(
            FETCH_CONTEXT,
            level=LEVEL,
            cid=cid,
            maxEntities=MAX_ENTITIES,
            maxChunks=MAX_CHUNKS,
        ).single()

    msg = llm.invoke(prompt.format_messages(
        level=LEVEL,
        cid=cid,
        entities=j(data["entities"]),
        chunks=j(data["chunks"]),
    ))
    txt = msg.content.strip()

    with driver.session(database=DATABASE) as session:
        session.run(UPDATE, level=LEVEL, cid=cid, txt=txt)

    return {"cid": cid, "len": len(txt)}

def main():
    driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
    driver.verify_connectivity()

    llm = ChatOpenAI(model=MODEL, temperature=0)

    with driver.session(database=DATABASE) as session:
        comms = session.run(FETCH_COMMUNITIES, level=LEVEL).data()
    cids = [r["cid"] for r in comms]

    print(f"Found {len(cids)} communities at level={LEVEL}. Summarizing...")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(summarize_one, driver, llm, cid) for cid in cids]
        for f in as_completed(futures):
            _ = f.result()
            done += 1
            if done % 25 == 0:
                print(f"Summarized {done}/{len(cids)}")

    driver.close()
    print(" Done.")

if __name__ == "__main__":
    main()
