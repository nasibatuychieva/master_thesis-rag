from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

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


# ---------------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmakg"

SCRIPT_NAME = "LlamaIndex_BM25_Hybrid_Retriever_Rerank"
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_short.jsonl"
)

# Chunk schema
CHUNK_LABEL = "Chunk"
TEXT_PROP = "text"          # Chunk.text
ID_PROP = "chunk_id"        # Chunk.chunk_id
FILE_PROP = "file"          # Chunk.file
PRODUCT_PROP = "product"
PRODCAT_PROP = "product_category"

# How many chunks to load for BM25 + Vector (RAM!)
CHUNK_LOAD_LIMIT = 80_000

# Retrieval sizes
BM25_TOP_K = 30
VECTOR_TOP_K = 30
PG_TOP_K = 30

# After merge
ENSEMBLE_TOP_K = 60         # candidates before rerank
FINAL_CONTEXT_K = 12        # what to pass to prompt & log

# Weights (optional)
W_BM25 = 1.0
W_VECTOR = 1.0
W_PG = 1.2  # give KG a slight boost

# Rerank (local)
USE_RERANK = True
RERANK_TOP_N = 12
RERANK_MODEL = "BAAI/bge-reranker-base"  # pip install sentence-transformers

# LLM + Embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)

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

# ---------------------------------------------------------------------------
# 2) Logging helper (context-aware)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3) Load Chunk nodes for BM25+Vector
# ---------------------------------------------------------------------------
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

        chunk_id = (r.get("chunk_id") or "").strip()
        meta = {
            "chunk_id": chunk_id,
            "file": r.get("file", ""),
            "product": r.get("product", ""),
            "product_category": r.get("product_category", ""),
            "retriever": "chunk_store",
            "node_type": "Chunk",  # <--- hilfreich fürs Logging
            "label": CHUNK_LABEL,  # <--- optional
        }

        n = TextNode(text=text, metadata=meta)
        if chunk_id:
            n.id_ = chunk_id
        nodes.append(n)

    print(f"[INFO] Loaded {len(nodes)} chunks for BM25+Vector.")
    return nodes


# ---------------------------------------------------------------------------
# 4) Ensemble Retriever (BM25 + Vector + PG)
# ---------------------------------------------------------------------------
class EnsembleRetriever(BaseRetriever):
    def __init__(
        self,
        bm25_retriever: Optional[BaseRetriever],
        vector_retriever: Optional[BaseRetriever],
        pg_retriever: Optional[BaseRetriever],
        *,
        weights: Dict[str, float],
        top_k: int,
        debug: bool = False,
    ):
        super().__init__()
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.pg = pg_retriever
        self.weights = weights
        self.top_k = top_k
        self.debug = debug

    def _dedupe_id(self, nws: NodeWithScore) -> str:
        node = nws.node
        # prefer chunk_id if present
        meta = getattr(node, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("chunk_id"):
            return str(meta["chunk_id"])
        # else id_
        nid = getattr(node, "id_", None)
        if nid:
            return str(nid)
        # else fallback
        txt = (node.get_content() or "")[:200]
        return f"txt:{hash(txt)}"

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        merged: List[NodeWithScore] = []
        seen = set()

        def add_nodes(name: str, nodes: List[NodeWithScore]):
            w = float(self.weights.get(name, 1.0))
            if self.debug:
                print(f"[DEBUG] {name}: {len(nodes)}")
            for nws in nodes:
                did = self._dedupe_id(nws)
                if did in seen:
                    continue
                seen.add(did)

                base_score = float(getattr(nws, "score", 0.0) or 0.0)
                merged.append(NodeWithScore(node=nws.node, score=base_score * w))

        if self.bm25 is not None:
            add_nodes("bm25", self.bm25.retrieve(query_bundle))
        if self.vector is not None:
            add_nodes("vector", self.vector.retrieve(query_bundle))
        if self.pg is not None:
            add_nodes("pg", self.pg.retrieve(query_bundle))

        merged.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return merged[: self.top_k]


# ---------------------------------------------------------------------------
# 5) Build QueryEngine once
# ---------------------------------------------------------------------------
QUERY_ENGINE: Optional[RetrieverQueryEngine] = None

def ensure_query_engine() -> RetrieverQueryEngine:
    global QUERY_ENGINE
    if QUERY_ENGINE is not None:
        return QUERY_ENGINE

    if BM25Retriever is None:
        raise ImportError(
            "BM25Retriever konnte nicht importiert werden.\n"
            "Versuche:\n"
            "  pip install -U llama-index llama-index-retrievers-bm25\n"
        )

    if USE_RERANK and SentenceTransformerRerank is None:
        raise ImportError(
            "SentenceTransformerRerank fehlt.\n"
            "Installiere:\n"
            "  pip install -U sentence-transformers\n"
        )

    print("[INFO] Building BM25+Vector indexes from Chunk nodes...")
    chunk_nodes = load_chunk_nodes(CHUNK_LOAD_LIMIT)

    # Dense index over chunks
    v_index = VectorStoreIndex(chunk_nodes, embed_model=embed_model)
    vector_retriever = VectorIndexRetriever(index=v_index, similarity_top_k=VECTOR_TOP_K)

    # Sparse BM25 over chunks
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

    # KG side
    kg_vector = VectorContextRetriever(
        graph_store=pg_index.property_graph_store,
        embed_model=embed_model,
        similarity_top_k=PG_TOP_K,
    )
    kg_syn = LLMSynonymRetriever(
        graph_store=pg_index.property_graph_store,
        llm=llm,
    )
    pg_retriever = PGRetriever(
        sub_retrievers=[kg_syn, kg_vector],
        llm=llm,
    )

    # Ensemble
    ensemble = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        pg_retriever=pg_retriever,
        weights={"bm25": W_BM25, "vector": W_VECTOR, "pg": W_PG},
        top_k=ENSEMBLE_TOP_K,
        debug=False,
    )

    node_postprocessors = []
    if USE_RERANK:
        node_postprocessors.append(
            SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL)
        )
    print("QA_PROMPT TEMPLATE:\n", QA_PROMPT.get_template())
    QUERY_ENGINE = RetrieverQueryEngine.from_args(
        retriever=ensemble,
        node_postprocessors=node_postprocessors,
        llm=llm,
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )
    print("[INFO] QueryEngine ready.")
        
    # prompts = QUERY_ENGINE.get_prompts()
    # for name, p in prompts.items():
    #     print(f"\n--- PROMPT: {name} ---\n")
    #     print(p.get_template())

    return QUERY_ENGINE


# ---------------------------------------------------------------------------
# 6) Ask + extract ONLY Chunk-text context_items for logging
# ---------------------------------------------------------------------------
def _is_chunk_node(meta: Dict[str, Any]) -> bool:
    """
    Decide whether this source node represents a Chunk.
    We keep this permissive enough to catch different LlamaIndex/Neo4j metadata shapes,
    but strict enough to exclude pure Entity/Relation nodes.
    """
    # strongest signal: chunk_id present
    if meta.get("chunk_id"):
        return True

    # common metadata keys across versions
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
    """
    Prefer node.get_content(), fallback to meta['text'] if needed.
    """
    txt = ""
    try:
        txt = (node.get_content() or "").strip()
    except Exception:
        txt = ""

    if not txt:
        txt = str(meta.get("text", "") or "").strip()

    return txt


def ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    qe = ensure_query_engine()
    resp = qe.query(question)
    answer_text = str(resp).strip()

    context_items: List[Dict[str, Any]] = []
    src_nodes = getattr(resp, "source_nodes", None) or []

    # IMPORTANT: log ONLY Chunk node texts
    for nws in src_nodes:
        node = nws.node
        meta = getattr(node, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}

        if not _is_chunk_node(meta):
            # skip KG-only nodes (Entity/Relation/Community etc.)
            continue

        content = _extract_chunk_text(node, meta)
        if not content:
            continue

        context_items.append(
            {
                "content": content,  # <- Chunk.text
                "node_type": "Chunk",  # <- so your logger counts it properly
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

    return answer_text, context_items


# ---------------------------------------------------------------------------
# 7) Batch + manual
# ---------------------------------------------------------------------------
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
    print("Ensemble: BM25(Chunk) + Vector(Chunk) + PGRetriever(KG) + Rerank")
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
