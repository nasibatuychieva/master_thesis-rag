import re

from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from pathlib import Path
from docling.chunking import HybridChunker
from PIL import Image
import pytesseract

# 
def tcount_func_factory(tokenizer):
    if hasattr(tokenizer, "count_tokens"):
        return lambda s: tokenizer.count_tokens(s)
    elif hasattr(tokenizer, "tok"):           
        return lambda s: len(tokenizer.tok.encode(s, add_special_tokens=False))
    elif hasattr(tokenizer, "tokenizer"):     
        return lambda s: len(tokenizer.tokenizer.encode(s, add_special_tokens=False))
    else:
        raise AttributeError("Tokenizer bietet keine count-Funktion an.")

def make_chunker(tokenizer):
    kwargs = dict(
        tokenizer=tokenizer,
        target_token_count=220,      
        max_tokens_per_chunk=260,    
        overlap_tokens=40,           
        merge_peers=False,          
        prefer_headings=True,        # an Headings schneiden
        join_short_paragraphs=False, 
    )

    return HybridChunker(**kwargs)

def convert_to_doc(md_text):

    with NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(md_text)
        md_path = f.name
    converter = DocumentConverter()
    result = converter.convert(md_path)
    doc_from_md = result.document 
    return doc_from_md
    
def split_by_tokens(text: str, tcount, target=220, maxlen=260):
    
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out, buf = [], ""
    for s in sents:
        if not s.strip():
            continue
  
        if tcount(s) > maxlen:
            words, cur = s.split(), []
            for w in words:
                cur.append(w)
                if tcount(" ".join(cur)) >= target:
                    out.append(" ".join(cur).strip()); cur = []
            if cur:
                out.append(" ".join(cur).strip())
            continue
   
        if not buf:
            buf = s
        else:
            test = f"{buf} {s}"
            if tcount(test) <= target:
                buf = test
            else:
                out.append(buf.strip()); buf = s
    if buf:
        out.append(buf.strip())


    final = []
    for ch in out:
        if tcount(ch) <= maxlen:
            final.append(ch)
        else:
            words, cur = ch.split(), []
            for w in words:
                cur.append(w)
                if tcount(" ".join(cur)) >= target:
                    final.append(" ".join(cur).strip()); cur = []
            if cur:
                final.append(" ".join(cur).strip())
    return [c for c in final if c]
