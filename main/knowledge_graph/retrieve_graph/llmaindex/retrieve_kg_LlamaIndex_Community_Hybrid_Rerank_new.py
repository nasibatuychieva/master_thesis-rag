from __future__ import annotations

import json
import os
import re
import time
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# BM25 import differs by version -> safe import
BM25Retriever = None
try:
    from llama_index.core.retrievers import BM25Retriever  # type: ignore
except Exception:
    try:
        from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore
    except Exception:
        BM25Retriever = None

# Local reranker (needs sentence-transformers)
SentenceTransformerRerank = None
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank  # type: ignore
except Exception:
    SentenceTransformerRerank = None

# KG / Property Graph
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
    PGRetriever,
)

from main.evaluation.logger import log_antwort


# =============================================================================
# 0) Generic utilities
# =============================================================================

STOPWORDS = {
    # minimal english stopwords (keep it lightweight, no nltk dependency)
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","for","with","by","as",
    "is","are","was","were","be","been","being","do","does","did","can","could","should","would","may","might",
    "i","you","we","they","he","she","it","my","your","our","their","this","that","these","those",
    "about","into","from","at","during","before","after","over","under","between","within","without",
    "what","which","why","how","where","who","whom"
}

NOISE_PREFIXES = (
    "Here are some facts extracted",
    "Here are facts extracted",
    "Facts extracted from the provided text",
)

def stable_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def retry_call(fn, *, tries: int = 5, base_sleep: float = 1.0, max_sleep: float = 20.0):
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(min(max_sleep, base_sleep * (2 ** i)))
    raise last_err

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_keywords(question: str, *, max_terms: int = 16) -> List[str]:
    q = normalize_text(question)
    toks = [t for t in q.split() if t and t not in STOPWORDS and len(t) >= 3]
    # keep order, de-duplicate
    seen = set()
    out: List[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out

def build_keyword_dense_query(question: str) -> str:
    kws = extract_keywords(question, max_terms=18)
    # fall back to original if keyword extraction fails
    if not kws:
        return question.strip()
    return " ".join(kws)

def is_noise_chunk(text: str) -> bool:
    t = (text or "").lstrip()
    return any(t.startswith(p) for p in NOISE_PREFIXES)

def context_adequacy_score(question: str, context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generic adequacy check based on keyword overlap.
    Returns dict with score + diagnostics.
    """
    kws = extract_keywords(question, max_terms=16)
    if not kws:
        return {"kws": [], "overlap": 0, "ratio": 0.0}

    ctx_blob = normalize_text(" ".join((c.get("content", "")[:1200] for c in context_items)))
    hit = sum(1 for k in kws if k in ctx_blob)

    ratio = hit / max(1, len(kws))
    return {"kws": kws, "overlap": hit, "ratio": ratio}

def should_retry_retrieval(adequacy: Dict[str, Any], *, min_ratio: float = 0.22) -> bool:
    """
    Generic threshold: if too few question keywords occur in context -> retrieval likely off-topic.
    Tune min_ratio if needed.
    """
    return float(adequacy.get("ratio") or 0.0) < min_ratio

def faithful_missing_info_answer(question: str, adequacy: Dict[str, Any]) -> str:
    kws = adequacy.get("kws") or []
    # Keep it short and evaluable
    if kws:
        return (
            "Missing information: The provided context does not contain enough relevant details to answer the question "
            "based solely on the documentation excerpts (low keyword overlap)."
        )
    return (
        "Missing information: The provided context does not contain enough relevant details to answer the question "
        "based solely on the documentation excerpts."
    )


def _extract_answer_from_resp(resp: Any) -> str:
    txt = getattr(resp, "response", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    msg = getattr(resp, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()

    txt2 = getattr(resp, "text", None)
    if isinstance(txt2, str) and txt2.strip():
        return txt2.strip()

    s = str(resp).strip()
    if s:
        return s
    return ""


# =============================================================================
# 1) Config
# =============================================================================
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmakg"

SCRIPT_NAME = "LlamaIndex_BM25_Vector_PG_Community_Rerank"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_short_empty_responses2.jsonl"
)

CHUNK_LABEL = "Chunk"
TEXT_PROP = "text"
ID_PROP = "chunk_id"
FILE_PROP = "file"
PRODUCT_PROP = "product"
PRODCAT_PROP = "product_category"

CHUNK_LOAD_LIMIT = 80_000

BM25_TOP_K = 30
VECTOR_TOP_K = 30
PG_TOP_K = 30

COMMUNITY_VEC_INDEX_NAME = os.getenv("COMMUNITY_VEC_INDEX_NAME", "community_vec")
COMMUNITY_LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))
COMMUNITY_TOP_K = 25
COMMUNITY_EXPAND_CHUNK_K = 250
COMMUNITY_CHUNK_TOP_K = 30

ENSEMBLE_TOP_K = 80
FINAL_CONTEXT_K = 12

W_BM25 = 1.0
W_VECTOR = 1.0
W_PG = 1.2
W_COMM = 0.8

USE_RERANK = True
RERANK_TOP_N = 12
RERANK_MODEL = "BAAI/bge-reranker-base"

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0, max_tokens=1200)

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

QA_PROMPT = PromptTemplate(
    "You are a technical support assistant for Arduino Products.\n"
    "Use ONLY the provided context. Do not use outside knowledge.\n"
    "If the context does not contain the answer, say exactly what information is missing.\n"
    "Answer in complete sentences.\n"
    "Answer as completely as possible.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question:\n"
    "{query_str}\n\n"
    "Answer:\n"
)

MAX_CHUNK_CHARS = 1800


# =============================================================================
# 2) Logging helper
# =============================================================================
def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: Optional[List[Dict[str, Any]]] = None,
):
    log_antwort(
        script,
        question_id,
        query_type,
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )


# =============================================================================
# 3) Load Chunk nodes for BM25+Vector (with noise filter)
# =============================================================================
def load_chunk_nodes(limit: int) -> List[TextNode]:
    cypher = f"""
    MATCH (c:{CHUNK_LABEL})
    WHERE c.{TEXT_PROP} IS NOT NULL AND trim(c.{TEXT_PROP}) <> ''
    RETURN
      coalesce(c.{ID_PROP}, '') AS chunk_id,
      coalesce(c.{FILE_PROP}, '') AS file,
      coalesce(c.{PRODUCT_PROP}, '') AS product,
      coalesce(c.{PRODCAT_PROP}, '') AS product_category,
      c.{TEXT_PROP} AS text
    LIMIT $limit
    """
    with driver.session(database=DATABASE) as s:
        rows = s.run(cypher, limit=limit).data()

    nodes: List[TextNode] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        # generic noise filter (important for your dataset!)
        if is_noise_chunk(text):
            continue

        chunk_id = (r.get("chunk_id") or "").strip()
        meta = {
            "chunk_id": chunk_id,
            "file": r.get("file", ""),
            "product": r.get("product", ""),
            "product_category": r.get("product_category", ""),
            "retriever": "chunk_store",
            "node_type": "Chunk",
            "label": CHUNK_LABEL,
        }

        n = TextNode(text=text, metadata=meta)
        if chunk_id:
            n.id_ = chunk_id
        nodes.append(n)

    print(f"[INFO] Loaded {len(nodes)} chunks for BM25+Vector.")
    return nodes


# =============================================================================
# 4) Community Router Retriever
# =============================================================================
class CommunityRouterRetriever(BaseRetriever):
    def __init__(
        self,
        driver,
        database: str,
        *,
        embed_model: OpenAIEmbedding,
        community_vec_index: str,
        community_level: int,
        top_k_communities: int,
        expand_chunk_k: int,
        chunk_top_k: int,
    ):
        super().__init__()
        self.driver = driver
        self.database = database
        self.embed_model = embed_model
        self.community_vec_index = community_vec_index
        self.community_level = int(community_level)
        self.top_k_communities = int(top_k_communities)
        self.expand_chunk_k = int(expand_chunk_k)
        self.chunk_top_k = int(chunk_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = str(query_bundle.query_str or "").strip()
        if not q:
            return []

        qvec = retry_call(lambda: self.embed_model.get_query_embedding(q), tries=4, base_sleep=0.8)
        if qvec is None:
            return []
        qvec = list(qvec)

        cy_comm_vec = """
        CALL db.index.vector.queryNodes($index_name, $k, $qvec)
        YIELD node, score
        WHERE node:__Community__
          AND (node.level IS NULL OR node.level = $level)
          AND node.embedding IS NOT NULL
        RETURN elementId(node) AS cid, score
        ORDER BY score DESC
        """

        with self.driver.session(database=self.database) as s:
            comm_rows = s.run(
                cy_comm_vec,
                index_name=self.community_vec_index,
                k=self.top_k_communities,
                qvec=qvec,
                level=self.community_level,
            ).data()

        community_ids = [r["cid"] for r in comm_rows if r.get("cid")]
        if not community_ids:
            return []

        cy_expand = f"""
        MATCH (c:__Community__)
        WHERE elementId(c) IN $community_ids
        MATCH (c)-[:IN_COMMUNITY]-(e:__Entity__)
        MATCH (e)-[:MENTIONS]-(ch:{CHUNK_LABEL})
        WHERE ch.{TEXT_PROP} IS NOT NULL AND trim(ch.{TEXT_PROP}) <> ''
        RETURN ch AS chunk, count(DISTINCT e) AS entity_hits
        ORDER BY entity_hits DESC
        LIMIT $k
        """

        with self.driver.session(database=self.database) as s:
            ch_rows = s.run(
                cy_expand,
                community_ids=community_ids,
                k=self.expand_chunk_k,
            ).data()

        out: List[NodeWithScore] = []
        for r in ch_rows:
            ch = r.get("chunk")
            hits = int(r.get("entity_hits") or 0)
            if ch is None:
                continue

            text = (ch.get(TEXT_PROP, "") or "").strip()
            if not text:
                continue
            if is_noise_chunk(text):
                continue

            chunk_id = str(ch.get(ID_PROP) or ch.get("id") or "").strip()
            meta = dict(ch)
            meta["chunk_id"] = chunk_id
            meta["retriever"] = "community"
            meta["node_type"] = "Chunk"
            meta["entity_hits"] = hits
            meta["label"] = CHUNK_LABEL

            n = TextNode(text=text, metadata=meta)
            if chunk_id:
                n.id_ = chunk_id
            out.append(NodeWithScore(node=n, score=float(hits)))

        out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return out[: self.chunk_top_k]


# =============================================================================
# 5) Ensemble Retriever (stable dedupe)
# =============================================================================
class EnsembleRetriever(BaseRetriever):
    def __init__(
        self,
        bm25_retriever: Optional[BaseRetriever],
        vector_retriever: Optional[BaseRetriever],
        pg_retriever: Optional[BaseRetriever],
        community_retriever: Optional[BaseRetriever],
        *,
        weights: Dict[str, float],
        top_k: int,
        debug: bool = False,
    ):
        super().__init__()
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.pg = pg_retriever
        self.community = community_retriever
        self.weights = weights
        self.top_k = top_k
        self.debug = debug

    def _dedupe_key(self, nws: NodeWithScore) -> str:
        node = nws.node
        meta = getattr(node, "metadata", None) or {}
        file_ = ""
        if isinstance(meta, dict):
            cid = str(meta.get("chunk_id") or "").strip()
            if cid:
                return f"chunk_id:{cid}"
            file_ = str(meta.get("file") or meta.get("source") or "").strip()
        try:
            txt = (node.get_content() or "").strip()
        except Exception:
            txt = ""
        return f"txtsha1:{file_}:{stable_sha1(txt[:2000])}"

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        merged: List[NodeWithScore] = []
        seen = set()

        def add_nodes(name: str, nodes: List[NodeWithScore]):
            w = float(self.weights.get(name, 1.0))
            if self.debug:
                print(f"[DEBUG] {name}: {len(nodes)}")
            for nws in nodes:
                key = self._dedupe_key(nws)
                if key in seen:
                    continue
                seen.add(key)
                base_score = float(getattr(nws, "score", 0.0) or 0.0)
                merged.append(NodeWithScore(node=nws.node, score=base_score * w))

        if self.bm25 is not None:
            add_nodes("bm25", self.bm25.retrieve(query_bundle))
        if self.vector is not None:
            add_nodes("vector", self.vector.retrieve(query_bundle))
        if self.pg is not None:
            add_nodes("pg", self.pg.retrieve(query_bundle))
        if self.community is not None:
            add_nodes("community", self.community.retrieve(query_bundle))

        merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return merged[: self.top_k]


# =============================================================================
# 6) Build QueryEngine once
# =============================================================================
QUERY_ENGINE: Optional[RetrieverQueryEngine] = None

def ensure_query_engine() -> RetrieverQueryEngine:
    global QUERY_ENGINE
    if QUERY_ENGINE is not None:
        return QUERY_ENGINE

    if BM25Retriever is None:
        raise ImportError("BM25Retriever missing. Install llama-index-retrievers-bm25.")

    if USE_RERANK and SentenceTransformerRerank is None:
        raise ImportError("SentenceTransformerRerank missing. Install sentence-transformers.")

    print("[INFO] Building BM25+Vector indexes from Chunk nodes...")
    chunk_nodes = load_chunk_nodes(CHUNK_LOAD_LIMIT)

    v_index = VectorStoreIndex(chunk_nodes, embed_model=embed_model)
    vector_retriever = VectorIndexRetriever(index=v_index, similarity_top_k=VECTOR_TOP_K)

    bm25_retriever = BM25Retriever.from_defaults(nodes=chunk_nodes, similarity_top_k=BM25_TOP_K)

    print("[INFO] Building PropertyGraphIndex / PGRetriever (KG)...")
    pg_store = Neo4jPGStore(
        username=AUTH_USER,
        password=AUTH_PASSWORD,
        url=URI,
        database=DATABASE,
    )
    pg_index = PropertyGraphIndex.from_existing(
        property_graph_store=pg_store,
        llm=llm,
        embed_model=embed_model,
    )

    kg_vector = VectorContextRetriever(
        graph_store=pg_index.property_graph_store,
        embed_model=embed_model,
        similarity_top_k=PG_TOP_K,
    )
    kg_syn = LLMSynonymRetriever(
        graph_store=pg_index.property_graph_store,
        llm=llm,
    )
    pg_retriever = PGRetriever(sub_retrievers=[kg_syn, kg_vector], llm=llm)

    community_retriever = CommunityRouterRetriever(
        driver=driver,
        database=DATABASE,
        embed_model=embed_model,
        community_vec_index=COMMUNITY_VEC_INDEX_NAME,
        community_level=COMMUNITY_LEVEL,
        top_k_communities=COMMUNITY_TOP_K,
        expand_chunk_k=COMMUNITY_EXPAND_CHUNK_K,
        chunk_top_k=COMMUNITY_CHUNK_TOP_K,
    )

    ensemble = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        pg_retriever=pg_retriever,
        community_retriever=community_retriever,
        weights={"bm25": W_BM25, "vector": W_VECTOR, "pg": W_PG, "community": W_COMM},
        top_k=ENSEMBLE_TOP_K,
        debug=False,
    )

    node_postprocessors = []
    if USE_RERANK:
        node_postprocessors.append(
            SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL)
        )

    QUERY_ENGINE = RetrieverQueryEngine.from_args(
        retriever=ensemble,
        node_postprocessors=node_postprocessors,
        llm=llm,
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )
    print("[INFO] QueryEngine ready.")
    return QUERY_ENGINE


# =============================================================================
# 7) Context extraction helpers
# =============================================================================
def _is_chunk_node(meta: Dict[str, Any]) -> bool:
    if meta.get("chunk_id"):
        return True
    nt = str(meta.get("node_type", "") or "").strip().lower()
    if nt == "chunk":
        return True
    label = str(meta.get("label", "") or "").strip().lower()
    if label == "chunk":
        return True
    return False

def _extract_chunk_text(node: Any, meta: Dict[str, Any]) -> str:
    txt = ""
    try:
        txt = (node.get_content() or "").strip()
    except Exception:
        txt = ""
    if not txt:
        txt = str(meta.get("text", "") or "").strip()
    if txt and len(txt) > MAX_CHUNK_CHARS:
        txt = txt[:MAX_CHUNK_CHARS] + "…"
    return txt


# =============================================================================
# 8) Ask: generic self-healing retrieval + faithful refusal
# =============================================================================
def _run_query_and_collect(qe: RetrieverQueryEngine, query_str: str) -> Tuple[str, List[Dict[str, Any]], Any]:
    resp = retry_call(lambda: qe.query(query_str), tries=5, base_sleep=1.0, max_sleep=16.0)
    answer_text = _extract_answer_from_resp(resp)

    context_items: List[Dict[str, Any]] = []
    src_nodes = getattr(resp, "source_nodes", None) or []

    for nws in src_nodes:
        node = nws.node
        meta = getattr(node, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        if not _is_chunk_node(meta):
            continue

        content = _extract_chunk_text(node, meta)
        if not content:
            continue
        if is_noise_chunk(content):
            continue

        context_items.append(
            {
                "content": content,
                "node_type": "Chunk",
                "source": meta.get("file", meta.get("source", "")),
                "id": meta.get("chunk_id") or getattr(node, "id_", "") or "",
                "score": float(getattr(nws, "score", 0.0) or 0.0),
                "product": meta.get("product", ""),
                "product_category": meta.get("product_category", ""),
                "retriever": meta.get("retriever", "ensemble"),
            }
        )
        if len(context_items) >= FINAL_CONTEXT_K:
            break

    return answer_text, context_items, resp


def ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    qe = ensure_query_engine()

    # ---- Round 1: original question
    try:
        ans1, ctx1, resp1 = _run_query_and_collect(qe, question)
    except Exception as e:
        return "Empty Response", [{"node_type": "Error", "error": repr(e), "trace": traceback.format_exc()}]

    adeq1 = context_adequacy_score(question, ctx1)
    ctx1.insert(0, {"node_type": "Debug", "phase": "round1", "adequacy": adeq1})

    # If we got a real answer, keep it
    if ans1.strip():
        return ans1.strip(), ctx1

    # If no answer, but context seems off-topic -> self-heal retrieval
    if should_retry_retrieval(adeq1):
        dense_q = build_keyword_dense_query(question)

        try:
            ans2, ctx2, resp2 = _run_query_and_collect(qe, dense_q)
        except Exception as e:
            ctx1.insert(0, {"node_type": "Error", "phase": "round2_failed", "error": repr(e), "trace": traceback.format_exc()})
            # faithful refusal is better than empty
            return faithful_missing_info_answer(question, adeq1), ctx1

        adeq2 = context_adequacy_score(question, ctx2)
        ctx2.insert(0, {"node_type": "Debug", "phase": "round2", "query_used": dense_q, "adequacy": adeq2})

        if ans2.strip():
            return ans2.strip(), ctx2

        # If still no answer, decide between refusal vs empty
        # If adequacy still low -> refusal
        if should_retry_retrieval(adeq2):
            return faithful_missing_info_answer(question, adeq2), ctx2

        # adequacy ok-ish but answer empty -> likely missing explicit statement
        # return explicit missing info instead of Empty Response
        return (
            "Missing information: The provided context appears related, but it does not contain an explicit statement "
            "or the necessary technical details to answer the question strictly from the excerpts.",
            ctx2,
        )

    # If context is not off-topic but still no answer -> missing explicit statement
    if ctx1:
        return (
            "Missing information: The provided context does not contain an explicit statement or enough technical detail "
            "to answer the question strictly from the excerpts.",
            ctx1,
        )

    # No context at all
    return "Empty Response", ctx1


# =============================================================================
# 9) Batch + manual
# =============================================================================
def run_batch_from_file() -> None:
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] dataset file not found.")
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
                SCRIPT_NAME,
                str(question_id),
                str(query_type),
                question,
                answer_text,
                gold_answer,
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
        SCRIPT_NAME,
        qid,
        qtype,
        question,
        answer_text,
        gold_answer,
        context_items=ctx_items,
    )


def main_loop() -> None:
    print("Ensemble: BM25(Chunk) + Vector(Chunk) + PGRetriever(KG) + CommunityRouter + Rerank")
    print("Type 'exit' to quit.\n")
    print(f"DB={DATABASE}")
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
