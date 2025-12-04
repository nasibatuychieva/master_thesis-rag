
from pathlib import Path
import re
import os
import json
import subprocess
from typing import Iterable, Optional
import importlib
import docling_chunker_tutorial_pdf_functions
importlib.reload(docling_chunker_tutorial_pdf_functions)
from prepare_tutorial_pdf_functions import ocr_pdf_with_tesseract, clean_html_extracted_pdf_text, to_markdown_from_sections, clean_text
from docling_chunker_tutorial_pdf_functions import  make_chunker, convert_to_doc, split_by_tokens, tcount_func_factory
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


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


def process_pdf(pdf_path: Path, out_dir: Path):
    
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
    
    raw_text = ocr_pdf_with_tesseract(pdf_path, dpi=300, lang="eng")
    cleaned_text = clean_html_extracted_pdf_text(raw_text)

    md_text = to_markdown_from_sections(cleaned_text)
    doc_from_md = convert_to_doc(md_text)

    tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
    max_tokens=260,    # an Max-Grenze
    )
    chunker = make_chunker(tokenizer)
    raw_chunks = list(chunker.chunk(dl_doc=doc_from_md))
    
    tcount = tcount_func_factory(tokenizer)
    
    tight_chunks = []
    
    for ch in raw_chunks:
        text_raw = (ch.text or "").strip()
        if not text_raw:
            continue
        parts = split_by_tokens(text_raw, tcount, target=220, maxlen=260)
        tight_chunks.extend(parts)

    # === 4) Optional: winzige Fragmente verwerfen oder an Nachbar anhängen

    MIN_WORDS = 25
    final_chunks = []
    buf = ""
    for c in tight_chunks:
        if len(c.split()) < MIN_WORDS:
        # hänge an vorherigen (falls vorhanden) an, ansonsten sammle
            if final_chunks:
                final_chunks[-1] = final_chunks[-1] + " " + c
            else:
                buf += (" " + c)
        else:
            if buf:
                final_chunks.append((buf + " " + c).strip()); buf = ""
            else:
                final_chunks.append(c)
    if buf:
        final_chunks.append(buf.strip())

# # === 5) Kontrolle/Stats
#     def tcount(x): return len(tokenizer.tok.encode(x, add_special_tokens=False))
#     sizes = [tcount(c) for c in final_chunks]
#     print(f"Chunks: {len(final_chunks)} | min/max/avg tokens: {min(sizes)}/{max(sizes)}/{sum(sizes)//len(sizes)}")

# === 6) Export (Beispiel)
    section = None
    n_tokens = None
    semantic_density = None

    records = []
    for i, text_raw in enumerate(final_chunks):
        records.append({
            "file_name": pdf_path.name,
            "chunk_id": f"{pdf_path.stem}::c{i}",
            "chunk_size": n_tokens,
            "semantic_density": semantic_density,
            "product_category": category,
            "product": product,
            "section": section,
            "tutorial": tutorial,
            "element": element,
            
            "text": f"[Product: {product}] [Product category: {category}] [Element of {product}: {element}] [Tutorial about {product}: {tutorial}] \n\n{text_raw}",
    })

    # === 2) HARTE NACHKONTROLLE: Alles, was trotzdem zu groß ist, erneut tokenbasiert splitten



    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] {len(records)} Chunks hinzugefügt zu: {out_path}")

def iterate_product_docs(
    doc_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
   
):
    # Root/Default-Pfade nur setzen, wenn nichts übergeben wurde
    if doc_root is None or out_dir is None:
        root = get_repo_root()
        parent_path = root.parent
        doc_root = doc_root or (parent_path / "documents")
        out_dir  = out_dir  or (parent_path / "out")

    out_dir.mkdir(parents=True, exist_ok=True)

    # load tokenizer

    for pdf_path in doc_root.rglob("*.pdf"):
        # ensure pdf_path is a file
        if not pdf_path.is_file():
            continue

        print(f"Start processing {pdf_path}")
        print(f"Start writing into {pdf_path.parent.parent.name} / {pdf_path.parent.name} / {pdf_path.name}")

        process_pdf(pdf_path, out_dir)


    





