from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

SentenceTransformerRerank = None
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank  # type: ignore
except Exception:
    SentenceTransformerRerank = None

from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever, PGRetriever

from main.evaluation.logger import log_antwort

# =============================================================================
# 1) CONFIG
# =============================================================================
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "llmakg")

SCRIPT_NAME = "LLmaIndex_Hybrid_KG_Retriever_Rerank_WITH_COMMUNITIES"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "golden_answers_dataset_new.jsonl"
)


print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
print("[DEBUG] QUESTIONS_PATH:", QUESTIONS_PATH)

# Chunk schema
NODE_LABEL = "Chunk"
TEXT_PROP = "text"
ID_PROP = "chunk_id"
FILE_PROP = "file"
PRODUCT_PROP = "product"
PRODCAT_PROP = "product_category"

# Community schema
COMM_LABEL = "__Community__"
COMM_ID_PROP = "communityId"
COMM_LEVEL_PROP = "level"
COMM_FULL_PROP = "full_content"
COMM_SUMMARY_PROP = "summary"
COMM_TOPIC_PROP = "topic_label"
COMM_RANK_PROP = "community_rank"
COMM_LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))

# Neo4j indexes (chunks)
FULLTEXT_INDEX_NAME = os.getenv("CHUNK_FT_INDEX", "chunk_text_ft")
VECTOR_INDEX_NAME = os.getenv("CHUNK_VEC_INDEX", "entity")

# Neo4j indexes (communities) - optional
COMMUNITY_FT_INDEX_NAME = os.getenv("COMMUNITY_FT_INDEX", "community_full_content_ft")
COMMUNITY_VEC_INDEX_NAME = os.getenv("COMMUNITY_VEC_INDEX", "communityEmbedding")

# Retrieval sizes
K_SPARSE = int(os.getenv("K_SPARSE", "30"))
K_DENSE = int(os.getenv("K_DENSE", "30"))
K_KG = int(os.getenv("K_KG", "15"))

K_COMM_FT = int(os.getenv("K_COMM_FT", "12"))
K_COMM_VEC = int(os.getenv("K_COMM_VEC", "12"))
K_COMM_FALLBACK = int(os.getenv("K_COMM_FALLBACK", "12"))

# Fusion weights
ALPHA = float(os.getenv("ALPHA", "0.60"))
W_KG = float(os.getenv("W_KG", "0.80"))
W_COMM = float(os.getenv("W_COMM", "0.55"))

# Candidate pool sizes
ENSEMBLE_TOP_K = int(os.getenv("ENSEMBLE_TOP_K", "100"))
FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", "10"))

# Rerank
USE_RERANK = os.getenv("USE_RERANK", "1") == "1"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "12"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Compacting for communities (avoid endless CSV)
COMPACT_COMM_MAX_ENTITIES = int(os.getenv("COMPACT_COMM_MAX_ENTITIES", "30"))
COMPACT_COMM_MAX_CHUNKS = int(os.getenv("COMPACT_COMM_MAX_CHUNKS", "6"))
COMPACT_COMM_MAX_CHUNK_CHARS = int(os.getenv("COMPACT_COMM_MAX_CHUNK_CHARS", "1200"))

embed_model = OpenAIEmbedding(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
llm = OpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=float(os.getenv("LLM_TEMPERATURE", "0")))

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# =============================================================================
# 2) LOGGING: make compact community content + safe_log (NEW LOGGER)
# =============================================================================
def make_compact_community_content(
    full_content: str,
    *,
    max_entities: int = COMPACT_COMM_MAX_ENTITIES,
    max_chunks: int = COMPACT_COMM_MAX_CHUNKS,
    max_chunk_chars: int = COMPACT_COMM_MAX_CHUNK_CHARS,
) -> str:
    s = (full_content or "").strip()
    if not s:
        return ""

    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
    except Exception:
        return s[: max_chunks * max_chunk_chars]

    lvl = obj.get("level", "")
    cid = obj.get("communityId", "")
    entities = obj.get("entities", []) or []
    top_products = obj.get("top_products", []) or []
    top_categories = obj.get("top_categories", []) or []
    chunks = obj.get("chunks", []) or []

    entities = [str(x) for x in entities[:max_entities]]

    def _fmt_top(lst: List[Dict[str, Any]], title: str, k: int = 10) -> str:
        items = []
        for it in lst[:k]:
            v = it.get("value", "")
            c = it.get("count", "")
            items.append(f"- {v} (count={c})")
        return title + "\n" + ("\n".join(items) if items else "- (none)")

    chunk_blocks = []
    for ch in chunks[:max_chunks]:
        prod = ch.get("product", "")
        cat = ch.get("product_category", "")
        chunk_id = ch.get("chunk_id", "")
        text = (ch.get("text", "") or "").strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + " ...[truncated]"
        chunk_blocks.append(f"[Chunk product={prod} category={cat} id={chunk_id}]\n{text}")

    out = []
    out.append(f"Community level={lvl} id={cid}")
    out.append("Entities (top): " + (", ".join(entities) if entities else "(none)"))
    out.append(_fmt_top(top_products, "Top products:"))
    out.append(_fmt_top(top_categories, "Top categories:"))
    if chunk_blocks:
        out.append("Evidence chunks:\n" + "\n\n".join(chunk_blocks))

    return "\n\n".join(out).strip()


def safe_log(
    script_name: str,
    question_id: Any,
    query_type: Any,
    question: str,
    answer: str,
    gold_answer: Any,
    context_items: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    IMPORTANT:
    - uses ONLY keyword args matching the new logger signature
    - never calls log_antwort with wrong kw names (prevents your previous TypeError)
    """
    try:
        log_antwort(
            script_name=str(script_name),
            question_id=str(question_id or ""),
            query_type=str(query_type or ""),
            question=str(question or ""),
            answer=str(answer or ""),
            gold_answer=str(gold_answer or ""),
            context_items=context_items or [],
        )
    except Exception as e:
        print("[WARN] logging failed:", repr(e))


# =============================================================================
# 3) UTIL
# =============================================================================
def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return [1.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 3]


def list_index_names(driver, database: str) -> set[str]:
    with driver.session(database=database) as session:
        try:
            rows = session.run("SHOW INDEXES YIELD name RETURN name").data()
            return {r["name"] for r in rows if r.get("name")}
        except Exception:
            pass

        for q in [
            "CALL db.indexes() YIELD name RETURN name",
            "CALL db.indexes YIELD name RETURN name",
        ]:
            try:
                rows = session.run(q).data()
                return {r["name"] for r in rows if r.get("name")}
            except Exception:
                continue
    return set()

def index_exists(driver, database: str, index_name: str) -> bool:
    if not index_name:
        return False
    return index_name in list_index_names(driver, database)


# =============================================================================
# 4) PRE-RETRIEVAL
# =============================================================================
def pre_retrieval(question: str) -> Dict[str, Any]:
    rewrite_prompt = f"""
Rewrite the following technical question to be concise, precise,
and unambiguous while preserving its meaning.

Question:
{question}
"""
    rewritten = llm.complete(rewrite_prompt).text.strip()

    expand_prompt = f"""
Given the following technical query:

{rewritten}

List 3–8 important keywords or short phrases (comma-separated, no explanations).
"""
    expansion = llm.complete(expand_prompt).text
    keywords = [k.strip() for k in expansion.split(",") if k.strip()]

    base = rewritten.replace("\n", " ").strip()
    base_terms = [t.strip() for t in base.split(" ") if t.strip()]
    base_terms = [t for t in base_terms if len(t) >= 2]

    base_part = " OR ".join([f'"{t}"' for t in base_terms[:12]])
    kw_part = " OR ".join([f'"{k}"^2' for k in keywords[:8]])

    if base_part and kw_part:
        ft_query = f"({base_part}) OR ({kw_part})"
    elif kw_part:
        ft_query = f"({kw_part})"
    elif base_part:
        ft_query = f"({base_part})"
    else:
        ft_query = '"arduino"'

    return {"original": question, "rewritten": rewritten, "keywords": keywords, "ft_query": ft_query}


# =============================================================================
# 5) NEO4J FULLTEXT RETRIEVER (Chunks)
# =============================================================================
class Neo4jFulltextRetriever(BaseRetriever):
    def __init__(self, driver, database: str, *, top_k: int):
        super().__init__()
        self.driver = driver
        self.database = database
        self.top_k = int(top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = str(query_bundle.query_str or "").strip()
        if not q:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $q)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """
        rows: List[Tuple[TextNode, float]] = []
        raw_scores: List[float] = []

        with self.driver.session(database=self.database) as session:
            res = session.run(cypher, index_name=FULLTEXT_INDEX_NAME, q=q, k=self.top_k)
            for r in res:
                node = r["node"]
                score = float(r["score"] or 0.0)
                text = (node.get(TEXT_PROP, "") or "").strip()
                if not text:
                    continue

                chunk_id = str(node.get(ID_PROP) or node.get("id") or "")
                meta = dict(node)
                meta["node_type"] = "chunk"
                meta["chunk_id"] = chunk_id
                meta["retriever"] = "sparse"

                n = TextNode(text=text, metadata=meta)
                if chunk_id:
                    n.id_ = chunk_id

                rows.append((n, score))
                raw_scores.append(score)

        norm = minmax_norm(raw_scores)
        return [NodeWithScore(node=n, score=float(ns)) for (n, _), ns in zip(rows, norm)]


# =============================================================================
# 6) NEO4J VECTOR RETRIEVER (Chunks)
# =============================================================================
class Neo4jVectorRetriever(BaseRetriever):
    def __init__(self, driver, database: str, *, embed_model: OpenAIEmbedding, top_k: int):
        super().__init__()
        self.driver = driver
        self.database = database
        self.embed_model = embed_model
        self.top_k = int(top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = str(query_bundle.query_str or "").strip()
        if not q:
            return []

        qvec = self.embed_model.get_query_embedding(q)

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $qvec)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        rows: List[Tuple[TextNode, float]] = []
        raw_scores: List[float] = []

        with self.driver.session(database=self.database) as session:
            res = session.run(cypher, index_name=VECTOR_INDEX_NAME, k=self.top_k, qvec=qvec)
            for r in res:
                node = r["node"]
                score = float(r["score"] or 0.0)
                text = (node.get(TEXT_PROP, "") or "").strip()
                if not text:
                    continue

                chunk_id = str(node.get(ID_PROP) or node.get("id") or "")
                meta = dict(node)
                meta["node_type"] = "chunk"
                meta["chunk_id"] = chunk_id
                meta["retriever"] = "dense"

                n = TextNode(text=text, metadata=meta)
                if chunk_id:
                    n.id_ = chunk_id

                rows.append((n, score))
                raw_scores.append(score)

        norm = minmax_norm(raw_scores)
        return [NodeWithScore(node=n, score=float(ns)) for (n, _), ns in zip(rows, norm)]


# =============================================================================
# 7) COMMUNITY RETRIEVAL
# =============================================================================
class CommunityRetriever(BaseRetriever):
    def __init__(
        self,
        driver,
        database: str,
        *,
        embed_model: OpenAIEmbedding,
        top_k_ft: int,
        top_k_vec: int,
        top_k_fallback: int,
        level: int,
    ):
        super().__init__()
        self.driver = driver
        self.database = database
        self.embed_model = embed_model
        self.top_k_ft = int(top_k_ft)
        self.top_k_vec = int(top_k_vec)
        self.top_k_fallback = int(top_k_fallback)
        self.level = int(level)

        self.has_ft = index_exists(driver, database, COMMUNITY_FT_INDEX_NAME)
        self.has_vec = index_exists(driver, database, COMMUNITY_VEC_INDEX_NAME)
        self._all_cache: Optional[List[Dict[str, Any]]] = None

    def _fetch_all(self) -> List[Dict[str, Any]]:
        if self._all_cache is not None:
            return self._all_cache
        cypher = f"""
        MATCH (c:{COMM_LABEL})
        WHERE c.{COMM_LEVEL_PROP} = $lvl
          AND c.{COMM_FULL_PROP} IS NOT NULL AND trim(c.{COMM_FULL_PROP}) <> ""
        RETURN
          c.{COMM_ID_PROP} AS communityId,
          c.{COMM_LEVEL_PROP} AS level,
          c.{COMM_FULL_PROP} AS full_content,
          c.{COMM_SUMMARY_PROP} AS summary,
          c.{COMM_TOPIC_PROP} AS topic_label,
          c.{COMM_RANK_PROP} AS community_rank
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(cypher, lvl=self.level).data()
        self._all_cache = rows
        return rows

    def _to_node(self, c: Dict[str, Any], *, score: float, channel: str) -> TextNode:
        full_content = (c.get("full_content") or "").strip()
        summary = (c.get("summary") or "").strip()
        topic = (c.get("topic_label") or "").strip()
        cid = str(c.get("communityId") or "")

        text = full_content if full_content else summary
        if topic and topic.lower() not in (text or "").lower():
            text = f"[Topic: {topic}]\n{text}"

        meta = {
            "node_type": "community",
            "communityId": cid,
            "level": c.get("level", self.level),
            "topic_label": topic,
            "community_rank": c.get("community_rank", None),
            "retriever": channel,
            "score_raw": score,
        }
        n = TextNode(text=text, metadata=meta)
        if cid:
            n.id_ = f"community:{cid}"
        return n

    def _retrieve_ft(self, q: str) -> List[NodeWithScore]:
        if not self.has_ft or not q:
            return []
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index_name, $q)
        YIELD node, score
        WITH node, score
        WHERE node:{COMM_LABEL} AND node.{COMM_LEVEL_PROP} = $lvl
        RETURN
          node.{COMM_ID_PROP} AS communityId,
          node.{COMM_LEVEL_PROP} AS level,
          node.{COMM_FULL_PROP} AS full_content,
          node.{COMM_SUMMARY_PROP} AS summary,
          node.{COMM_TOPIC_PROP} AS topic_label,
          node.{COMM_RANK_PROP} AS community_rank,
          score AS score
        ORDER BY score DESC
        LIMIT $k
        """
        rows: List[Tuple[TextNode, float]] = []
        raw_scores: List[float] = []
        with self.driver.session(database=self.database) as session:
            res = session.run(
                cypher,
                index_name=COMMUNITY_FT_INDEX_NAME,
                q=q,
                k=self.top_k_ft,
                lvl=self.level,
            )
            for r in res:
                d = dict(r)
                sc = float(d.get("score") or 0.0)
                n = self._to_node(d, score=sc, channel="community_ft")
                rows.append((n, sc))
                raw_scores.append(sc)

        norm = minmax_norm(raw_scores)
        return [NodeWithScore(node=n, score=float(ns)) for (n, _), ns in zip(rows, norm)]

    def _retrieve_vec(self, q: str) -> List[NodeWithScore]:
        if not self.has_vec or not q:
            return []
        qvec = self.embed_model.get_query_embedding(q)
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $qvec)
        YIELD node, score
        WITH node, score
        WHERE node:{COMM_LABEL} AND node.{COMM_LEVEL_PROP} = $lvl
        RETURN
          node.{COMM_ID_PROP} AS communityId,
          node.{COMM_LEVEL_PROP} AS level,
          node.{COMM_FULL_PROP} AS full_content,
          node.{COMM_SUMMARY_PROP} AS summary,
          node.{COMM_TOPIC_PROP} AS topic_label,
          node.{COMM_RANK_PROP} AS community_rank,
          score AS score
        ORDER BY score DESC
        LIMIT $k
        """
        rows: List[Tuple[TextNode, float]] = []
        raw_scores: List[float] = []
        with self.driver.session(database=self.database) as session:
            res = session.run(
                cypher,
                index_name=COMMUNITY_VEC_INDEX_NAME,
                k=self.top_k_vec,
                qvec=qvec,
                lvl=self.level,
            )
            for r in res:
                d = dict(r)
                sc = float(d.get("score") or 0.0)
                n = self._to_node(d, score=sc, channel="community_vec")
                rows.append((n, sc))
                raw_scores.append(sc)

        norm = minmax_norm(raw_scores)
        return [NodeWithScore(node=n, score=float(ns)) for (n, _), ns in zip(rows, norm)]

    def _retrieve_fallback(self, q: str) -> List[NodeWithScore]:
        all_cs = self._fetch_all()
        if not all_cs:
            return []
        q_toks = tokens(q)
        if not q_toks:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for c in all_cs:
            t = ((c.get("full_content") or "") + " " + (c.get("summary") or "")).lower()
            s = 0.0
            for w in q_toks:
                if w in t:
                    s += 1.0
            rnk = c.get("community_rank")
            if rnk is not None:
                try:
                    s += 0.05 * float(rnk)
                except Exception:
                    pass
            if s > 0:
                scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k_fallback]

        raw_scores = [float(s) for s, _ in top]
        norm = minmax_norm(raw_scores)

        out: List[NodeWithScore] = []
        for (s, c), ns in zip(top, norm):
            n = self._to_node(c, score=float(s), channel="community_fallback")
            out.append(NodeWithScore(node=n, score=float(ns)))
        return out

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = str(query_bundle.query_str or "").strip()
        if not q:
            return []
        ft = self._retrieve_ft(q)
        vec = self._retrieve_vec(q)
        if ft or vec:
            merged: Dict[str, Dict[str, Any]] = {}
            for nws in ft + vec:
                nid = getattr(nws.node, "id_", "") or nws.node.metadata.get("communityId", "")
                key = str(nid)
                if key not in merged:
                    merged[key] = {"node": nws.node, "ft": None, "vec": None}
                if nws.node.metadata.get("retriever") == "community_ft":
                    merged[key]["ft"] = float(nws.score or 0.0)
                if nws.node.metadata.get("retriever") == "community_vec":
                    merged[key]["vec"] = float(nws.score or 0.0)

            out: List[NodeWithScore] = []
            for obj in merged.values():
                score = max(float(obj["ft"] or 0.0), float(obj["vec"] or 0.0))
                out.append(NodeWithScore(node=obj["node"], score=float(score)))
            out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
            return out[: max(self.top_k_ft, self.top_k_vec)]

        return self._retrieve_fallback(q)


# =============================================================================
# 8) KG RETRIEVER + normalization wrapper
# =============================================================================
class KGNormalizedRetriever(BaseRetriever):
    def __init__(self, kg_retriever: BaseRetriever, *, top_k: int):
        super().__init__()
        self.kg_retriever = kg_retriever
        self.top_k = int(top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        items = self.kg_retriever.retrieve(query_bundle)
        items = items[: self.top_k] if items else []
        if not items:
            return []

        raw_scores = [float(nws.score or 0.0) for nws in items]
        norm = minmax_norm(raw_scores)

        out: List[NodeWithScore] = []
        for nws, ns in zip(items, norm):
            node = nws.node
            meta = dict(getattr(node, "metadata", {}) or {})
            meta["node_type"] = meta.get("node_type", "chunk")
            meta["retriever"] = "kg"
            if "chunk_id" not in meta:
                meta["chunk_id"] = meta.get(ID_PROP) or getattr(node, "id_", "") or ""
            new_node = TextNode(text=node.get_content(), metadata=meta)
            if meta.get("chunk_id"):
                new_node.id_ = str(meta["chunk_id"])
            out.append(NodeWithScore(node=new_node, score=float(ns)))
        return out


# =============================================================================
# 9) ENSEMBLE
# =============================================================================
class HybridKGCommunityEnsembleRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        sparse: BaseRetriever,
        dense: BaseRetriever,
        kg: BaseRetriever,
        comm: BaseRetriever,
        alpha: float,
        w_kg: float,
        w_comm: float,
        top_k: int,
    ):
        super().__init__()
        self.sparse = sparse
        self.dense = dense
        self.kg = kg
        self.comm = comm
        self.alpha = float(alpha)
        self.w_kg = float(w_kg)
        self.w_comm = float(w_comm)
        self.top_k = int(top_k)

    def _key(self, node: TextNode) -> str:
        meta = getattr(node, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("node_type") == "community":
            cid = str(meta.get("communityId") or "")
            return f"community:{cid}" if cid else f"community_txt:{hash((node.get_content() or '')[:200])}"
        cid = ""
        if isinstance(meta, dict):
            cid = str(meta.get("chunk_id") or meta.get(ID_PROP) or "")
        if cid:
            return f"chunk:{cid}"
        nid = getattr(node, "id_", None)
        if nid:
            return f"node:{nid}"
        txt = (node.get_content() or "")[:200]
        return f"txt:{hash(txt)}"

    def _merge_channel(self, merged: Dict[str, Dict[str, Any]], channel: str, items: List[NodeWithScore]) -> None:
        for nws in items:
            node = nws.node
            key = self._key(node)
            score = float(nws.score or 0.0)

            if key not in merged:
                meta = dict(getattr(node, "metadata", {}) or {})
                meta.setdefault("retrievers", [])
                if channel not in meta["retrievers"]:
                    meta["retrievers"].append(channel)

                if meta.get("node_type") == "community":
                    meta.setdefault("communityId", meta.get(COMM_ID_PROP, ""))
                    meta.setdefault("level", meta.get(COMM_LEVEL_PROP, COMM_LEVEL))
                else:
                    meta.setdefault("chunk_id", meta.get(ID_PROP) or getattr(node, "id_", "") or "")
                    meta.setdefault("file", meta.get(FILE_PROP, ""))
                    meta.setdefault("product", meta.get(PRODUCT_PROP, ""))
                    meta.setdefault("product_category", meta.get(PRODCAT_PROP, ""))

                new_node = TextNode(text=node.get_content(), metadata=meta)
                if meta.get("node_type") == "community" and meta.get("communityId"):
                    new_node.id_ = f"community:{meta['communityId']}"
                elif meta.get("chunk_id"):
                    new_node.id_ = str(meta["chunk_id"])

                merged[key] = {"node": new_node, "sparse": None, "dense": None, "kg": None, "comm": None}

            if channel == "sparse":
                merged[key]["sparse"] = score
            elif channel == "dense":
                merged[key]["dense"] = score
            elif channel == "kg":
                merged[key]["kg"] = score
            elif channel == "comm":
                merged[key]["comm"] = score

            meta2 = merged[key]["node"].metadata
            meta2.setdefault("retrievers", [])
            if channel not in meta2["retrievers"]:
                meta2["retrievers"].append(channel)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        original_q = str(query_bundle.query_str or "").strip()
        if not original_q:
            return []

        pre = pre_retrieval(original_q)
        q_dense = pre["rewritten"]
        q_sparse = pre["ft_query"]

        merged: Dict[str, Dict[str, Any]] = {}

        sparse_items = self.sparse.retrieve(QueryBundle(q_sparse))
        dense_items = self.dense.retrieve(QueryBundle(q_dense))
        kg_items = self.kg.retrieve(QueryBundle(q_dense))
        comm_items = self.comm.retrieve(QueryBundle(q_dense))

        self._merge_channel(merged, "sparse", sparse_items)
        self._merge_channel(merged, "dense", dense_items)
        self._merge_channel(merged, "kg", kg_items)
        self._merge_channel(merged, "comm", comm_items)

        out: List[NodeWithScore] = []
        for obj in merged.values():
            s = float(obj["sparse"] or 0.0)
            d = float(obj["dense"] or 0.0)
            k = float(obj["kg"] or 0.0)
            c = float(obj["comm"] or 0.0)

            hybrid = (self.alpha * d) + ((1.0 - self.alpha) * s)
            final_score = hybrid + (self.w_kg * k) + (self.w_comm * c)

            meta = obj["node"].metadata
            meta["sparse_norm"] = obj["sparse"]
            meta["dense_norm"] = obj["dense"]
            meta["kg_norm"] = obj["kg"]
            meta["comm_norm"] = obj["comm"]
            meta["hybrid_score"] = hybrid
            meta["final_score"] = final_score
            meta["rewritten_query"] = pre["rewritten"]
            meta["ft_query"] = pre["ft_query"]

            out.append(NodeWithScore(node=obj["node"], score=float(final_score)))

        out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return out[: self.top_k]


# =============================================================================
# 10) BUILD QUERY ENGINE ONCE
# =============================================================================
QUERY_ENGINE: Optional[RetrieverQueryEngine] = None

def ensure_query_engine() -> RetrieverQueryEngine:
    global QUERY_ENGINE
    if QUERY_ENGINE is not None:
        return QUERY_ENGINE

    if USE_RERANK and SentenceTransformerRerank is None:
        raise ImportError("SentenceTransformerRerank fehlt. pip install -U sentence-transformers")

    print("[INFO] Building Neo4j sparse+dense retrievers (FULLTEXT + VECTOR)...")
    sparse = Neo4jFulltextRetriever(driver, DATABASE, top_k=K_SPARSE)
    dense = Neo4jVectorRetriever(driver, DATABASE, embed_model=embed_model, top_k=K_DENSE)

    print("[INFO] Building PropertyGraphIndex / KG retriever...")
    pg_store = Neo4jPGStore(username=AUTH_USER, password=AUTH_PASSWORD, url=URI, database=DATABASE)
    pg_index = PropertyGraphIndex.from_existing(property_graph_store=pg_store, llm=llm, embed_model=embed_model)

    kg_vector = VectorContextRetriever(
        graph_store=pg_index.property_graph_store,
        embed_model=embed_model,
        similarity_top_k=K_KG,
    )
    kg_base = PGRetriever(sub_retrievers=[kg_vector], llm=llm)
    kg = KGNormalizedRetriever(kg_base, top_k=K_KG)

    print("[INFO] Building Community retriever...")
    comm = CommunityRetriever(
        driver,
        DATABASE,
        embed_model=embed_model,
        top_k_ft=K_COMM_FT,
        top_k_vec=K_COMM_VEC,
        top_k_fallback=K_COMM_FALLBACK,
        level=COMM_LEVEL,
    )

    print("[INFO] Building Hybrid+KG+Community ensemble retriever...")
    ensemble = HybridKGCommunityEnsembleRetriever(
        sparse=sparse,
        dense=dense,
        kg=kg,
        comm=comm,
        alpha=ALPHA,
        w_kg=W_KG,
        w_comm=W_COMM,
        top_k=ENSEMBLE_TOP_K,
    )

    node_postprocessors = []
    if USE_RERANK:
        node_postprocessors.append(SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL))

    QUERY_ENGINE = RetrieverQueryEngine.from_args(
        retriever=ensemble,
        node_postprocessors=node_postprocessors,
        llm=llm,
    )
    print("[INFO] QueryEngine ready.")
    return QUERY_ENGINE


# =============================================================================
# 11) ASK + EXTRACT CONTEXT ITEMS FOR LOGGING (KEY FIX)
# =============================================================================
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _detect_node_type(meta: Dict[str, Any]) -> str:
    nt = str(meta.get("node_type", "") or "").strip().lower()
    if nt:
        return nt
    if meta.get("communityId") is not None or meta.get("level") is not None:
        return "community"
    return "chunk"

def ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    qe = ensure_query_engine()
    resp = qe.query(question)
    answer_text = str(resp).strip()

    src_nodes = getattr(resp, "source_nodes", None) or []
    context_items: List[Dict[str, Any]] = []

    for nws in src_nodes[:FINAL_CONTEXT_K]:
        node = getattr(nws, "node", None)
        if node is None:
            continue

        meta = _safe_dict(getattr(node, "metadata", None))

        # content
        try:
            content = (node.get_content() or "").strip()
        except Exception:
            content = str(getattr(node, "text", "") or "").strip()
        if not content:
            continue

        node_type = _detect_node_type(meta)

        # IMPORTANT: for community -> log compact content, NOT huge full_content
        if node_type == "community":
            content_for_log = make_compact_community_content(content)
        else:
            content_for_log = content

        base: Dict[str, Any] = {
            "content": content_for_log,
            "score": _to_float(getattr(nws, "score", 0.0), 0.0),
            "retrievers": meta.get("retrievers", meta.get("retriever", "unknown")),
            "sparse_norm": meta.get("sparse_norm", None),
            "dense_norm": meta.get("dense_norm", None),
            "kg_norm": meta.get("kg_norm", None),
            "comm_norm": meta.get("comm_norm", None),
            "hybrid_score": meta.get("hybrid_score", None),
            "final_score": meta.get("final_score", None),
            "node_type": node_type,
        }

        node_id_fallback = ""
        for attr in ("id_", "node_id", "id"):
            if hasattr(node, attr):
                node_id_fallback = str(getattr(node, attr) or "")
                if node_id_fallback:
                    break

        if node_type == "community":
            community_id = meta.get("communityId", "") or node_id_fallback
            context_items.append(
                {
                    **base,
                    "id": str(community_id),
                    "source": f"community:{community_id}",
                    "communityId": meta.get("communityId", community_id),
                    "level": meta.get("level", COMM_LEVEL),
                    "topic_label": meta.get("topic_label", meta.get(COMM_TOPIC_PROP, "")),
                    "community_rank": meta.get("community_rank", None),
                }
            )
        else:
            context_items.append(
                {
                    **base,
                    "id": str(meta.get("chunk_id") or node_id_fallback),
                    "source": meta.get(FILE_PROP, meta.get("source", "")),
                    "product": meta.get(PRODUCT_PROP, ""),
                    "product_category": meta.get(PRODCAT_PROP, ""),
                }
            )

    return answer_text, context_items


# =============================================================================
# 12) BATCH + MANUAL
# =============================================================================
def run_batch_from_file() -> None:
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset_new.jsonl not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] Invalid JSON at line {line_no}, skipped.")
                continue

            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id") or ""
            query_type = obj.get("query_type") or ""
            question = (obj.get("question") or "").strip()
            gold_answer = obj.get("gold_answer") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer_text, ctx_items = ask(question)
            print(f"[ANSWER] {answer_text}\n")

            safe_log(
                script_name=SCRIPT_NAME,
                question_id=question_id,
                query_type=query_type,
                question=question,
                answer=answer_text,
                gold_answer=gold_answer,
                context_items=ctx_items,
            )

    print("\n[INFO] Batch processing completed.\n")

def manual_question() -> None:
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional): ").strip() or ""
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or ""

    answer_text, ctx_items = ask(question)
    print("\nAnswer:\n", answer_text)

    safe_log(
        script_name=SCRIPT_NAME,
        question_id=qid,
        query_type=qtype,
        question=question,
        answer=answer_text,
        gold_answer=gold_answer,
        context_items=ctx_items,
    )

def main_loop() -> None:
    print("Hybrid: Neo4j FULLTEXT + Neo4j VECTOR + KG (PropertyGraph) + Communities + Local Rerank")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n): ").strip().lower()
        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question()
        elif mode in ("n", "no"):
            run_batch_from_file()
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

if __name__ == "__main__":
    try:
        main_loop()
    finally:
        driver.close()
