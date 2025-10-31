# -*- coding: utf-8 -*-
import os, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ---------------- Root-Ermittlung ----------------
def resolve_root() -> Path:
    candidates: List[Path] = []
    if os.environ.get("MAIN_ROOT"):
        candidates.append(Path(os.environ["MAIN_ROOT"]))
    if getattr(sys, "argv", None) and sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve().parent)
    candidates.append(Path(os.getcwd()))
    for c in candidates:
        for base in [c, *c.parents]:
            m = base / "main"
            if (m / "documents").exists():
                return m
    return Path(os.getcwd()) / "main"

# ---------------- Tokenizer ----------------
import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")
def count_tokens(s: str) -> int:
    return len(ENC.encode(s or ""))

# ---------------- Dataklasse ----------------
@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    meta: Dict[str, Any]

# ---------------- Parser ----------------
from docling.document_converter import DocumentConverter
def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    conv = DocumentConverter()
    res = conv.convert(pdf_path)
    out: List[Dict[str, Any]] = []
    for el in res.document.elements:
        et = getattr(el, "category", None) or getattr(el, "type", None) or "paragraph"
        out.append({
            "type": str(et).lower(),
            "text": getattr(el, "text", None),
            "page": getattr(el, "page", None),
            "section_path": getattr(el, "section_path", None) or [],
            "table_markdown": getattr(el, "markdown", None) if str(et).lower() == "table" else None,
        })
    return out

def parse_html(path: str) -> List[Dict[str, Any]]:
    from unstructured.partition.html import partition_html
    elements = partition_html(filename=path)
    return [{
        "type": (getattr(e, "category", None) or e.__class__.__name__).lower(),
        "text": getattr(e, "text", None),
        "page": None,
        "section_path": [],
        "table_markdown": None
    } for e in elements]

# ---------------- Unified Chunker (Fallback/HTML) ----------------
def unified_chunk(elements: List[Dict[str, Any]], doc_id: str,
                  token_budget: int = 1000) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf_texts: List[str] = []
    buf_meta: List[Dict[str, Any]] = []
    buf_tokens = 0

    def flush():
        nonlocal buf_texts, buf_meta, buf_tokens
        if not buf_texts:
            return
        txt = "\n".join(buf_texts).strip()
        cid = f"{doc_id}::c{len(chunks)}"
        meta = {
            "doc_id": doc_id,
            "pages": sorted({m.get("page") for m in buf_meta if m.get("page") is not None}),
            "section_path": buf_meta[-1].get("section_path", []) if buf_meta else [],
        }
        chunks.append(Chunk(cid, doc_id, txt, meta))
        buf_texts.clear(); buf_meta.clear(); buf_tokens = 0

    for el in elements:
        et = (el.get("type") or "").lower()
        if "table" in et or "code" in et:
            flush()
            content = el.get("table_markdown") or el.get("text") or ""
            if content.strip():
                cid = f"{doc_id}::c{len(chunks)}"
                meta = {
                    "doc_id": doc_id,
                    "element_type": et,
                    "pages": [el.get("page")] if el.get("page") is not None else [],
                    "section_path": el.get("section_path", []),
                }
                chunks.append(Chunk(cid, doc_id, content, meta))
        else:
            t = el.get("text") or ""
            tl = count_tokens(t)
            if buf_tokens and buf_tokens + tl > token_budget:
                flush()
            buf_texts.append(t)
            buf_meta.append(el)
            buf_tokens += tl

    flush()
    return chunks

# ---------------- Docling Hybrid Adapter (PDF) ----------------
def docling_hybrid_chunk(elements: List[Dict[str, Any]], doc_id: str,
                         token_budget: int = 1000) -> List[Chunk]:
    # Versuch: offizieller Hybrid-Chunker
    try:
        from docling.chunking import HybridChunker
        dl = HybridChunker(max_tokens=token_budget, overlap_tokens=150)
        dl_chunks = dl.chunk(elements)
        return [Chunk(f"{doc_id}::c{i}", doc_id, c.get("text") or "", c.get("meta", {}))
                for i, c in enumerate(dl_chunks)]
    except Exception:
        # Fallback auf unified (gleiche Ausgabeform)
        return unified_chunk(elements, doc_id, token_budget)

# ---------------- Discovery & IO ----------------
def discover_docs(doc_root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in doc_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in {".pdf", ".html", ".htm"}:
            continue
        parts = p.relative_to(doc_root).parts
        fam = parts[0] if len(parts) > 0 else None
        prod = parts[1] if len(parts) > 1 else None
        doc_type = "datasheet" if ext == ".pdf" else "tutorial"
        items.append({"path": str(p), "ext": ext, "family": fam, "product": prod, "doc_type": doc_type})
    return items

def save_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- main ----------------
def main():
    root = resolve_root()
    doc_root = root / "documents"
    out_file = root / "out" / "chunks_docling.jsonl"

    if not doc_root.exists():
        raise RuntimeError(f"documents-Ordner nicht gefunden: {doc_root}")

    items = discover_docs(doc_root)
    if not items:
        print(f"Keine Dateien unter {doc_root}")
        return

    all_chunks: List[Chunk] = []
    for it in items:
        doc_path = it["path"]
        ext = it["ext"]
        doc_id = os.path.relpath(doc_path, root)

        if ext == ".pdf":
            elements = parse_pdf(doc_path)
            chunks = docling_hybrid_chunk(elements, doc_id, token_budget=1000)
        else:
            elements = parse_html(doc_path)
            chunks = unified_chunk(elements, doc_id, token_budget=1000)

        all_chunks.extend(chunks)
        print(f"[OK] {doc_path} -> {len(chunks)}")


    save_jsonl(out_file, [c.__dict__ for c in all_chunks])
    print(f"✅ {len(all_chunks)} chunks → {out_file}")

if __name__ == "__main__":
    main()
