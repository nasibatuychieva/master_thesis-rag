# -*- coding: utf-8 -*-
import os, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ---------------- Root-Ermittlung (ohne __file__) ----------------
def resolve_root() -> Path:
    """
    Robust in Databricks Script Tasks:
    - 1) Umgebungsvariable MAIN_ROOT (optional)
    - 2) sys.argv[0] (Pfad zum Script)
    - 3) os.getcwd()
    Danach wird nach .../main mit einem 'documents'-Ordner gesucht.
    """
    candidates: List[Path] = []
    env_root = os.environ.get("MAIN_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    if getattr(sys, "argv", None) and sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve().parent)

    candidates.append(Path(os.getcwd()))

    for c in candidates:
        for base in [c, *c.parents]:
            m = base / "main"
            if (m / "documents").exists():
                return m
    # letzter Fallback: cwd/main (wird ggf. gleich bemängelt)
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

# ---------------- Parser: PDF (Docling) ----------------
from typing import Any, Dict, List

def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extrahiert Docling-Elemente versionstolerant (flache Liste aus Baum oder elements-Attribut)."""
    from docling.document_converter import DocumentConverter

    conv = DocumentConverter()
    res = conv.convert(pdf_path)
    doc = getattr(res, "document", None)
    out: List[Dict[str, Any]] = []

    def extract_elements(node):
        # rekursive Extraktion
        if node is None:
            return
        # Text oder Tabelle identifizieren
        etype = getattr(node, "category", None) or getattr(node, "type", None) or "paragraph"
        text = getattr(node, "text", None)
        page = getattr(node, "page", None)
        section_path = getattr(node, "section_path", None) or []
        table_md = getattr(node, "markdown", None) if str(etype).lower() == "table" else None
        if text:
            out.append({
                "type": str(etype).lower(),
                "text": text,
                "page": page,
                "section_path": section_path if isinstance(section_path, list) else [section_path],
                "table_markdown": table_md
            })
        # Kinder-Elemente weiter traversieren
        for child in getattr(node, "children", []) or []:
            extract_elements(child)

    # ✅ Variante 1: neue Struktur mit root
    root = getattr(doc, "root", None)
    if root is not None:
        extract_elements(root)
    else:
        # ✅ Variante 2: ältere Struktur mit elements
        elements = getattr(doc, "elements", None) or getattr(res, "elements", None)
        if elements:
            for el in elements:
                extract_elements(el)
        else:
            print(f"[WARN] Keine Elemente gefunden für {pdf_path}")

    return out

# ---------------- Parser: HTML (Unstructured) ----------------
def parse_html_with_unstructured(path: str) -> List[Dict[str, Any]]:
    from unstructured.partition.html import partition_html
    elements = partition_html(filename=path)
    out: List[Dict[str, Any]] = []
    for el in elements:
        et = getattr(el, "category", None) or el.__class__.__name__
        out.append({
            "type": str(et).lower(),
            "text": getattr(el, "text", None),
            "page": None,
            "section_path": [],
            "table_markdown": None,
        })
    return out

# ---------------- Einheitlicher Hybrid-Chunker ----------------
def hybrid_chunk(elements: List[Dict[str, Any]], doc_id: str,
                 token_budget: int = 1000) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf_texts: List[str] = []
    buf_meta: List[Dict[str, Any]] = []
    buf_tokens = 0

    def flush():
        nonlocal buf_texts, buf_meta, buf_tokens
        if not buf_texts:
            return
        text = "\n".join(buf_texts).strip()
        cid = f"{doc_id}::c{len(chunks)}"
        meta = {
            "doc_id": doc_id,
            "pages": sorted({m.get("page") for m in buf_meta if m.get("page") is not None}),
            "section_path": buf_meta[-1].get("section_path", []) if buf_meta else [],
        }
        chunks.append(Chunk(cid, doc_id, text, meta))
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
        family = parts[0] if len(parts) > 0 else None
        product = parts[1] if len(parts) > 1 else None
        doc_type = "datasheet" if ext == ".pdf" else "tutorial"
        tutorial_type = None
        if "tutorials" in p.parts:
            try:
                i = p.parts.index("tutorials")
                if len(p.parts) > i + 1:
                    tutorial_type = p.parts[i + 1]
            except ValueError:
                pass
        items.append({
            "path": str(p),
            "ext": ext,
            "doc_type": doc_type,
            "product_family": family,
            "product": product,
            "tutorial_type": tutorial_type,
        })
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
    out_file = root / "out" / "chunks_unified.jsonl"

    if not doc_root.exists():
        raise RuntimeError(f"documents-Ordner nicht gefunden: {doc_root}")

    items = discover_docs(doc_root)
    if not items:
        print(f"Keine Dateien unter {doc_root}")
        return

    all_chunks: List[Chunk] = []
    for it in items:
        doc_id = os.path.relpath(it["path"], root)
        base_meta = {
            "source": "pdf" if it["ext"] == ".pdf" else "html",
            "doc_type": it["doc_type"],
            "product_family": it["product_family"],
            "product": it["product"],
            "tutorial_type": it["tutorial_type"],
            "rel_path": os.path.relpath(it["path"], root),
        }
        try:
            if it["ext"] == ".pdf":
                elements = parse_pdf_with_docling(it["path"])
            else:
                elements = parse_html_with_unstructured(it["path"])
            chunks = hybrid_chunk(elements, doc_id, token_budget=1000)
            for c in chunks:
                c.meta.update(base_meta)
            all_chunks.extend(chunks)
            print(f"[OK] {it['path']} -> {len(chunks)}")
        except Exception as e:
            print(f"[ERROR] {it['path']}: {e}")

    save_jsonl(out_file, [c.__dict__ for c in all_chunks])
    print(f"✅ {len(all_chunks)} chunks → {out_file}")

if __name__ == "__main__":
    main()
