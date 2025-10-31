from pathlib import Path
import os
import subprocess
from typing import Iterable, Optional
from clean_pdf_functions import clean_text, should_drop_chunk
import importlib
import docling_chunker_functions
from pathlib import Path
import json
import prepare_html_functions 
importlib.reload(docling_chunker_functions)
importlib.reload(prepare_html_functions)
from docling_chunker_functions import convert_documents_into_docling_doc, chunk_documents_with_docling, return_tokenizer
from prepare_html_functions import build_docling_from_html

def get_repo_root(
    start_path: Optional[Path] = None,
    markers: Optional[Iterable[str]] = None
) -> Path:

    start = Path(start_path) if start_path else Path.cwd()
    markers = list(markers) if markers else [
        '.git', 'pyproject.toml', 'setup.cfg', 'requirements.txt'
    ]

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip())
    except Exception:
        pass 
   
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        for m in markers:
            if (parent / m).exists():
                return parent
    print(Path(os.getcwd()))
    return Path(os.getcwd())

def process_pdf(pdf_path: Path, out_dir: Path, doc, chunker, tokenizer):

    category = pdf_path.parent.parent.name
    product  = pdf_path.parent.name
    filename = pdf_path.stem  # Dateiname ohne .pdf
    parts = filename.split("_")  # ["Elements", "Bluetooth", "Espressif", "ESP32-C3-MINI-1U"]
    element = None
    if parts and parts[0] == "Elements":
        element = "_".join(parts[:2])
    tutorial = None
    if parts and parts[0] == "Tutorial":
        tutorial = "_".join(parts[:2])
    
    out_path = out_dir / category / "docling_chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    raw_chunks = list(chunker.chunk(dl_doc=doc))
    total_chunks = len(raw_chunks)

    records = []
    for i, ch in enumerate(raw_chunks):
        text_raw = clean_text(ch.text or "")
        if len(text_raw) < 30:
            continue

        context = clean_text(chunker.contextualize(chunk=ch))
        if len(context.split()) < 25:
            continue

        if should_drop_chunk(ch, context): 
            continue

        # section (defensiv)
        section = None
        hp = getattr(ch, "hierarchy_path", None)
        if isinstance(hp, list) and hp:
            last = hp[-1]
            if isinstance(last, dict):
                section = last.get("title")

        n_tokens = tokenizer.count_tokens(context)
        semantic_density = round(n_tokens / max(1, len(context)), 4)

        rec = {
            "category": category,
            "chunk_id": f"{pdf_path.stem}::c{i}",
            "chunk_size": n_tokens,
            "chunk_type": "contextualized",
            "product": product,
            "element": element,
            "tutorial": tutorial,
            "section": section,
            "semantic_density": semantic_density,
            "text": f"[Product: {product}] [Category: {category}] [Element of {product}: {element}] [Tutorial: {tutorial}] \n\n{context}",
            "total_chunks": total_chunks,
        }
        records.append(rec)

    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] {len(records)} Chunks hinzugefügt zu: {out_path}")

def iterate_product_docs(
    doc_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    doc=None, chunker=None, tokenizer=None
):
    # Root/Default-Pfade nur setzen, wenn nichts übergeben wurde
    if doc_root is None or out_dir is None:
        root = get_repo_root()
        parent_path = root.parent
        doc_root = doc_root or (parent_path / "documents")
        out_dir  = out_dir  or (parent_path / "out")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load tokenizer
    tokenizer = tokenizer or return_tokenizer()

    for pdf_path in doc_root.rglob("*.html"):
        # ensure pdf_path is a file
        if not pdf_path.is_file():
            continue

        print(f"Start processing {pdf_path}")
        print(f"Start writing into {pdf_path.parent.parent.name} / {pdf_path.parent.name} / {pdf_path.name}")
        
        #generate for each file doc 
        doc = build_docling_from_html(pdf_path)
        chunker_for_doc = chunk_documents_with_docling(doc, tokenizer) if chunker is None else chunker

        process_pdf(pdf_path, out_dir, doc, chunker_for_doc, tokenizer)


    





