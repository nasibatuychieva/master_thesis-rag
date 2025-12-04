import json
from pathlib import Path
from typing import Iterator, List, Dict
from langchain_core.documents import Document
from itertools import islice
from typing import Iterable, List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.chat_models import ChatLlamaCpp
from langchain_experimental.graph_transformers import LLMGraphTransformer

def stream_jsonl(jsonl_path: str | Path) -> Iterator[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def load_docs_from_jsonl(jsonl_path: str | Path,
                         text_key: str = "text",
                         meta_keys: List[str] = ("chunk_id","product","category","chunk_type","chunk_size","semantic_density","total_chunks")
                        ) -> List[Document]:
    docs: List[Document] = []
    for row in stream_jsonl(jsonl_path):
        txt = (row.get(text_key) or "").strip()
        if not txt:
            continue
        meta = {k: row.get(k) for k in meta_keys if k in row}
        docs.append(Document(page_content=txt, metadata=meta))
    return docs


chat_llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",  # dein Pfad
    temperature=0,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,  # CPU-only
)

gt = LLMGraphTransformer(
    llm=chat_llm,
    # optional: allowed_nodes=["Product","Interface","Feature","Family"],
    # optional: allowed_relationships=[("Product","has_interface","Interface"), ...]
)


def batched(iterable: Iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def docs_to_graph_docs(docs: List[Document], batch_size: int = 8) -> List[GraphDocument]:
    all_graph_docs: List[GraphDocument] = []
    for batch in batched(docs, batch_size):
        # convert_to_graph_documents ist synchron; falls du asynchron willst: aconvert_to_graph_documents
        gds = gt.convert_to_graph_documents(batch)
        all_graph_docs.extend(gds)
    return all_graph_docs
