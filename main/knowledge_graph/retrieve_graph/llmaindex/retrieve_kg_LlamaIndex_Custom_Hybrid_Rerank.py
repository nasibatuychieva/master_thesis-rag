from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

# LlamaIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Local reranker (sentence-transformers)
SentenceTransformerRerank = None
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank  # type: ignore
except Exception:
    SentenceTransformerRerank = None

# KG / Property Graph
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever, PGRetriever

from main.evaluation.logger import log_antwort


# =============================================================================
# 1) CONFIG
# =============================================================================
load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")

DATABASE = "llmakg"

SCRIPT_NAME = "LlamaIndex_Custom_Hybrid_Retriever_Rerank"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "golden_answers_dataset_short.jsonl"
)

# Chunk schema (Neo4j)
NODE_LABEL = "Chunk"
TEXT_PROP = "text"
ID_PROP = "chunk_id"  # stable id
FILE_PROP = "file"
PRODUCT_PROP = "product"
PRODCAT_PROP = "product_category"

# Neo4j indexes
FULLTEXT_INDEX_NAME = "chunk_text_ft"
VECTOR_INDEX_NAME = "entity"
EMB_PROPERTY = "embedding"

# Retrieval sizes
K_SPARSE = 30
K_DENSE = 30
K_KG = 15

# Fusion weights
ALPHA = 0.60
W_KG = 0.80

# Candidate pool sizes
ENSEMBLE_TOP_K = 60     # candidates BEFORE rerank
FINAL_CONTEXT_K = 12

# Rerank
USE_RERANK = True
RERANK_TOP_N = 12
RERANK_MODEL = "BAAI/bge-reranker-base"
USE_PRE_RETRIEVAL = False 
# LLM + Embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0, max_tokens=1200)

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()



QA_PROMPT = PromptTemplate(
    "You are a technical support assistant for Arduino Products.\n"
    "Use ONLY the provided context. Do not use outside knowledge.\n"
    "If the context does not contain the answer, say exactly what information is missing.\n"
    "Answer in complete sentences.\n"
    "Answer as completely as possible.\n"
    "Adapt the structure and style of the answer to the type of the question "
    "(e.g., list items for 'which' questions, explain processes for 'how' questions, "
    "and compare variants for 'difference' questions).\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question:\n"
    "{query_str}\n\n"
    "Answer:\n"
)

# =============================================================================
# 2) LOGGING HELPER
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
    # Einmal versuchen – und wenn es knallt, soll es sichtbar sein.
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
# 3) UTIL: MINMAX NORMALIZATION
# =============================================================================
def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return [1.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


# =============================================================================
# 4) PRE-RETRIEVAL (Rewrite + Keywords + Fulltext query string)
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

    return {
        "original": question,
        "rewritten": rewritten,
        "keywords": keywords,
        "ft_query": ft_query,
    }


# =============================================================================
# 5) NEO4J FULLTEXT RETRIEVER (Sparse)
# =============================================================================
class Neo4jFulltextRetriever(BaseRetriever):
    def __init__(self, driver, database: str, *, top_k: int):
        super().__init__()
        self.driver = driver
        self.database = database
        self.top_k = top_k

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
            res = session.run(
                cypher,
                index_name=FULLTEXT_INDEX_NAME,
                q=q,
                k=self.top_k,
            )
            for r in res:
                node = r["node"]
                score = float(r["score"] or 0.0)
                text = (node.get(TEXT_PROP, "") or "").strip()
                if not text:
                    continue

                chunk_id = str(node.get(ID_PROP) or node.get("id") or "")
                meta = dict(node)
                meta["chunk_id"] = chunk_id
                meta["retriever"] = "sparse"
                meta["node_type"] = "Chunk"  # <--- NEW

                n = TextNode(text=text, metadata=meta)
                if chunk_id:
                    n.id_ = chunk_id

                rows.append((n, score))
                raw_scores.append(score)

        # normalize
        norm = minmax_norm(raw_scores)
        out: List[NodeWithScore] = []
        for (n, _), ns in zip(rows, norm):
            out.append(NodeWithScore(node=n, score=float(ns)))
        return out


# =============================================================================
# 6) NEO4J VECTOR RETRIEVER (Dense)
# =============================================================================
class Neo4jVectorRetriever(BaseRetriever):
    def __init__(self, driver, database: str, *, embed_model: OpenAIEmbedding, top_k: int):
        super().__init__()
        self.driver = driver
        self.database = database
        self.embed_model = embed_model
        self.top_k = top_k

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
            res = session.run(
                cypher,
                index_name=VECTOR_INDEX_NAME,
                k=self.top_k,
                qvec=qvec,
            )
            for r in res:
                node = r["node"]
                score = float(r["score"] or 0.0)
                text = (node.get(TEXT_PROP, "") or "").strip()
                if not text:
                    continue

                chunk_id = str(node.get(ID_PROP) or node.get("id") or "")
                meta = dict(node)
                meta["chunk_id"] = chunk_id
                meta["retriever"] = "dense"
                meta["node_type"] = "Chunk"  # <--- NEW

                n = TextNode(text=text, metadata=meta)
                if chunk_id:
                    n.id_ = chunk_id

                rows.append((n, score))
                raw_scores.append(score)

        norm = minmax_norm(raw_scores)
        out: List[NodeWithScore] = []
        for (n, _), ns in zip(rows, norm):
            out.append(NodeWithScore(node=n, score=float(ns)))
        return out


# =============================================================================
# 7) KG RETRIEVER (Property Graph) + normalization wrapper
# =============================================================================
class KGNormalizedRetriever(BaseRetriever):
    """
    Wraps a KG retriever and normalizes its scores to 0..1 for stable fusion.
    Ensures node_type is set so the logger can count types.
    """
    def __init__(self, kg_retriever: BaseRetriever, *, top_k: int):
        super().__init__()
        self.kg_retriever = kg_retriever
        self.top_k = top_k

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
            meta["retriever"] = "kg"
            if "chunk_id" not in meta:
                meta["chunk_id"] = meta.get(ID_PROP) or getattr(node, "id_", "") or ""

            # if we do have chunk_id, treat it as Chunk context
            meta.setdefault("node_type", "Chunk" if meta.get("chunk_id") else "KG")

            new_node = TextNode(text=node.get_content() or "", metadata=meta)
            if meta.get("chunk_id"):
                new_node.id_ = str(meta["chunk_id"])
            out.append(NodeWithScore(node=new_node, score=float(ns)))

        return out


# =============================================================================
# 8) FUSION RETRIEVER (Sparse + Dense + KG), dedupe = MERGE (not drop)
# =============================================================================
class HybridKGEnsembleRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        sparse: BaseRetriever,
        dense: BaseRetriever,
        kg: BaseRetriever,
        alpha: float,
        w_kg: float,
        top_k: int,
    ):
        super().__init__()
        self.sparse = sparse
        self.dense = dense
        self.kg = kg
        self.alpha = float(alpha)
        self.w_kg = float(w_kg)
        self.top_k = int(top_k)

    def _key(self, node: TextNode) -> str:
        meta = getattr(node, "metadata", None) or {}
        cid = ""
        if isinstance(meta, dict):
            cid = str(meta.get("chunk_id") or meta.get(ID_PROP) or "")
        if cid:
            return cid
        nid = getattr(node, "id_", None)
        if nid:
            return str(nid)
        txt = (node.get_content() or "")[:200]
        return f"txt:{hash(txt)}"

    def _merge_channel(
        self,
        merged: Dict[str, Dict[str, Any]],
        channel: str,
        items: List[NodeWithScore],
    ) -> None:
        for nws in items:
            node = nws.node
            key = self._key(node)
            score = float(nws.score or 0.0)

            if key not in merged:
                meta = dict(getattr(node, "metadata", {}) or {})
                meta.setdefault("retrievers", [])
                meta["retrievers"].append(channel)

                meta.setdefault("chunk_id", meta.get(ID_PROP) or getattr(node, "id_", "") or "")
                meta.setdefault("file", meta.get(FILE_PROP, ""))
                meta.setdefault("product", meta.get(PRODUCT_PROP, ""))
                meta.setdefault("product_category", meta.get(PRODCAT_PROP, ""))

                # make sure logger sees chunk type
                meta.setdefault("node_type", "Chunk")

                new_node = TextNode(text=node.get_content() or "", metadata=meta)
                if meta.get("chunk_id"):
                    new_node.id_ = str(meta["chunk_id"])

                merged[key] = {
                    "node": new_node,
                    "sparse": None,
                    "dense": None,
                    "kg": None,
                }

            if channel == "sparse":
                merged[key]["sparse"] = score
            elif channel == "dense":
                merged[key]["dense"] = score
            elif channel == "kg":
                merged[key]["kg"] = score

            meta2 = merged[key]["node"].metadata
            meta2.setdefault("retrievers", [])
            if channel not in meta2["retrievers"]:
                meta2["retrievers"].append(channel)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        original_q = str(query_bundle.query_str or "").strip()
        if not original_q:
            return []

        if USE_PRE_RETRIEVAL:
            pre = pre_retrieval(original_q)
            q_dense = pre["rewritten"]
            q_sparse = pre["ft_query"]
        else:
            pre = {
        "original": original_q,
        "rewritten": original_q,
        "keywords": [],
        "ft_query": original_q,
        }
            q_dense = original_q
            q_sparse = original_q

        merged: Dict[str, Dict[str, Any]] = {}

        sparse_items = self.sparse.retrieve(QueryBundle(q_sparse))
        dense_items = self.dense.retrieve(QueryBundle(q_dense))
        kg_items = self.kg.retrieve(QueryBundle(q_dense))

        self._merge_channel(merged, "sparse", sparse_items)
        self._merge_channel(merged, "dense", dense_items)
        self._merge_channel(merged, "kg", kg_items)

        out: List[NodeWithScore] = []
        for obj in merged.values():
            s = float(obj["sparse"] or 0.0)
            d = float(obj["dense"] or 0.0)
            k = float(obj["kg"] or 0.0)

            hybrid = (self.alpha * d) + ((1.0 - self.alpha) * s)
            final_score = hybrid + (self.w_kg * k)

            meta = obj["node"].metadata
            meta["sparse_norm"] = obj["sparse"]
            meta["dense_norm"] = obj["dense"]
            meta["kg_norm"] = obj["kg"]
            meta["hybrid_score"] = hybrid
            meta["final_score"] = final_score
            meta["rewritten_query"] = pre["rewritten"]
            meta["ft_query"] = pre["ft_query"]

            out.append(NodeWithScore(node=obj["node"], score=float(final_score)))

        out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return out[: self.top_k]


# =============================================================================
# 9) BUILD QUERY ENGINE ONCE
# =============================================================================
QUERY_ENGINE: Optional[RetrieverQueryEngine] = None

# =============================================================================
# X) CHUNK FILTER (pre-rerank)
# =============================================================================
def _is_chunk_node(meta: Dict[str, Any]) -> bool:
    # strong signal
    if meta.get("chunk_id"):
        return True

    nt = str(meta.get("node_type", "") or "").strip().lower()
    if nt == "chunk":
        return True

    label = str(meta.get("label", "") or "").strip().lower()
    if label == "chunk":
        return True

    labels = meta.get("labels")
    if isinstance(labels, (list, tuple, set)):
        if any(str(x).strip().lower() == "chunk" for x in labels):
            return True

    return False


def _extract_chunk_text(node: Any, meta: Dict[str, Any]) -> str:
    txt = ""
    try:
        txt = (node.get_content() or "").strip()
    except Exception:
        txt = ""

    if not txt:
        txt = str(meta.get("text", "") or meta.get(TEXT_PROP, "") or "").strip()

    return txt


# LlamaIndex postprocessor interface
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

class ChunkOnlyPostprocessor(BaseNodePostprocessor):
    """Keep only nodes that look like Chunk nodes and have non-empty text."""
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        out: List[NodeWithScore] = []
        for nws in nodes:
            node = nws.node
            meta = getattr(node, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}

            if not _is_chunk_node(meta):
                continue

            content = _extract_chunk_text(node, meta)
            if not content:
                continue

            # ensure node content is actually the chunk text
            # (some KG nodes may have empty node text but meta text)
            if not (node.get_content() or "").strip():
                # replace with real TextNode content
                new_node = TextNode(text=content, metadata=meta)
                if meta.get("chunk_id"):
                    new_node.id_ = str(meta["chunk_id"])
                out.append(NodeWithScore(node=new_node, score=float(nws.score or 0.0)))
            else:
                out.append(nws)

        return out


def ensure_query_engine() -> RetrieverQueryEngine:
    global QUERY_ENGINE
    if QUERY_ENGINE is not None:
        return QUERY_ENGINE

    if USE_RERANK and SentenceTransformerRerank is None:
        raise ImportError(
            "SentenceTransformerRerank fehlt.\n"
            "Installiere:\n"
            "  pip install -U sentence-transformers\n"
        )

    print("[INFO] Building Neo4j sparse+dense retrievers (FULLTEXT + VECTOR)...")
    sparse = Neo4jFulltextRetriever(driver, DATABASE, top_k=K_SPARSE)
    dense = Neo4jVectorRetriever(driver, DATABASE, embed_model=embed_model, top_k=K_DENSE)

    print("[INFO] Building PropertyGraphIndex / KG retriever...")
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
        similarity_top_k=K_KG,
    )
    kg_base = PGRetriever(sub_retrievers=[kg_vector], llm=llm)
    kg = KGNormalizedRetriever(kg_base, top_k=K_KG)

    print("[INFO] Building Hybrid+KG ensemble retriever...")
    ensemble = HybridKGEnsembleRetriever(
        sparse=sparse,
        dense=dense,
        kg=kg,
        alpha=ALPHA,
        w_kg=W_KG,
        top_k=ENSEMBLE_TOP_K,
    )

    node_postprocessors = []

    # 1) First: remove non-chunk / empty-text nodes (pre-rerank!)
    node_postprocessors.append(ChunkOnlyPostprocessor())

    # 2) Then: rerank only real chunk candidates
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
# 10) ASK + EXTRACT CONTEXT ITEMS FOR LOGGING
# =============================================================================
def ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    qe = ensure_query_engine()
    resp = qe.query(question)
    answer_text = str(resp).strip()

    context_items: List[Dict[str, Any]] = []
    src_nodes = getattr(resp, "source_nodes", None) or []

    for nws in src_nodes[:FINAL_CONTEXT_K]:
        node = nws.node
        meta = getattr(node, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}

        # --- ROBUST: always log chunk text as "content" ---
        content = (node.get_content() or "").strip()
        if not content:
            # fallback: sometimes metadata carries the actual chunk text
            content = str(meta.get(TEXT_PROP, "") or "").strip()
        if not content:
            continue

        context_items.append(
            {
                "content": content,  # <- chunk text
                "node_type": meta.get("node_type", "Chunk"),  # <- NEW (for logger types)
                "source": meta.get(FILE_PROP, meta.get("source", "")),
                "id": meta.get("chunk_id") or getattr(node, "id_", "") or "",
                "score": float(getattr(nws, "score", 0.0) or 0.0),
                "product": meta.get(PRODUCT_PROP, ""),
                "product_category": meta.get(PRODCAT_PROP, ""),
                "retrievers": meta.get("retrievers", meta.get("retriever", "ensemble")),
                "sparse_norm": meta.get("sparse_norm", None),
                "dense_norm": meta.get("dense_norm", None),
                "kg_norm": meta.get("kg_norm", None),
                "hybrid_score": meta.get("hybrid_score", None),
                "final_score": meta.get("final_score", None),
            }
        )

    return answer_text, context_items


# =============================================================================
# 11) BATCH + MANUAL
# =============================================================================
def run_batch_from_file() -> None:
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset.jsonl not found.")
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

            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id")
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
                question_id,
                query_type,
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
    print("Hybrid: Neo4j FULLTEXT + Neo4j VECTOR + KG (PropertyGraph) + Local Rerank")
    print("Type 'exit' to quit.\n")
    print(f"DB={DATABASE} | FULLTEXT={FULLTEXT_INDEX_NAME} | VECTOR={VECTOR_INDEX_NAME} (prop {EMB_PROPERTY})")
    print(f"Fusion: alpha={ALPHA}, w_kg={W_KG} | K: sparse={K_SPARSE}, dense={K_DENSE}, kg={K_KG}")
    print(f"EnsembleTopK={ENSEMBLE_TOP_K} | RerankTopN={RERANK_TOP_N} | FinalContextK={FINAL_CONTEXT_K}\n")

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
