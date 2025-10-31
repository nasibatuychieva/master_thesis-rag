import re
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter
from pathlib import Path

def build_docling_from_html(html_path: Path):
    """
    Lädt eine HTML-Datei, bereinigt sie (Code, Boilerplate etc.)
    und gibt ein DoclingDocument-Objekt zurück.
    """
    # 1️⃣ HTML laden und säubern
    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "lxml")

    # Noise entfernen
    for tag in soup(["script","style","noscript","header","footer","nav","aside","form","svg"]):
        tag.decompose()

    # Codeblöcke in Markdown-Fences
    for pre in soup.find_all("pre"):
        code = pre.get_text("\n", strip=True)
        pre.string = f"\n```\n{code}\n```\n"

    # Inline-Code markieren
    for c in soup.find_all("code"):
        txt = c.get_text(" ", strip=True)
        c.string = f"`{txt}`"

    # Absätze/Listen normalisieren
    for el in soup.find_all(["p","li"]):
        if el.string:
            el.string.replace_with(re.sub(r"\s+\n\s+", "\n", el.string))

    clean_html = str(soup)

    # 2️⃣ Temporäre Datei schreiben & mit Docling konvertieren
    with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        tmp.write(clean_html)
        tmp_path = tmp.name

    converter = DocumentConverter()
    doc = converter.convert(tmp_path).document

    return doc
