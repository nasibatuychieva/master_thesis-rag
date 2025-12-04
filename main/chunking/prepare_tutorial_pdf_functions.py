import re

from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter
from pathlib import Path
from paddleocr import PaddleOCR
import fitz  # PyMuPDF

from PIL import Image
import pytesseract

# Optical Character Recognition 
def ocr_pdf_with_tesseract(pdf_path: str, dpi: int = 300, lang: str = "eng") -> str:
    doc = fitz.open(pdf_path)
    out = []
    for page in doc:
        # 1) Normale Textschicht versuchen
        txt = page.get_text("text").strip()
        if txt:
            out.append(txt)
            continue
        # 2) Sonst OCR auf gerenderter Seite
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        out.append(pytesseract.image_to_string(img, lang=lang))
    doc.close()
    return "\n\n".join(out).strip()


# 2.1 Cleaning 
def clean_html_extracted_pdf_text(t: str) -> str:
        # >>>> ERSETZE bei Bedarf durch deine Version mit Ligaturen etc. <<<<
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        t = t.replace("\r", "")
        t = re.sub(r"[ \t]{2,}", " ", t)
        # Navigations-/Müllzeilen optional herausfiltern
        t = re.sub(r"^\s*(Go Back|ON THIS PAGE|Author.*|Last revision.*|Help|Arduino\s*Docs)\s*$",
                   "", t, flags=re.I|re.M)
        t = re.sub(r"\n{2,}", "\n", t)
        return t.strip()

# 1) pull OCR-/PDF together
def normalize_paragraphs(text: str) -> str:
    lines = text.splitlines()
    out, buf = [], ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            if buf:
                out.append(buf); buf = ""
            continue
        if buf and not buf.endswith(('.', '!', '?', ':')):
            buf += " " + ln
        else:
            if buf: out.append(buf)
            buf = ln
    if buf: out.append(buf)
    return "\n".join(out)

# 2) Heading
def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s.split()) < 2:  # mind. 2 Wörter
        return False
    if len(s) > 80:
        return False
    if any(ch in ".!?;:" for ch in s):  # keine Sätze mit Punkt etc.
        return False
    # grobe Heuristik: ALL CAPS oder Title Case
    return s.isupper() or s.istitle()

# 3) Semantic Sections 
def split_into_sections_semantic(text: str):
    txt = normalize_paragraphs(text)
    sections, cur_title, cur_buf = [], "", []
    for ln in txt.splitlines():
        if is_heading(ln):
            if cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
                cur_buf = []
            cur_title = ln.strip()
        else:
            cur_buf.append(ln)
    if cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
    # Fallback: all in one section
    return sections or [("", text.strip())]

# Create Markdown for Docling-Import
def to_markdown_from_sections(cleaned_text: str) -> str:
    md_lines = []
    for title, body in split_into_sections_semantic(cleaned_text):
        if title:
            md_lines.append(f"# {title}")
        md_lines.append(body)
        md_lines.append("")  # Leerzeile
    return "\n".join(md_lines).strip()

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = t.replace("\r", "")
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    def noisy(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        non_alpha = sum(1 for ch in s if not ch.isalpha())
        return (non_alpha / max(1, len(s))) > 0.6

    lines = [ln for ln in t.split("\n") if not noisy(ln)]
    t = "\n".join(lines)
    t = re.sub(r"^(Table|Figure)\s*\d+[:.\-]\s.*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*\|.*\|\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-=]{3,}\s*$", "", t, flags=re.MULTILINE)
    return t.strip()