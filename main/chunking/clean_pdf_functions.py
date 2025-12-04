import re
# Cleaning-Funktion
# ---------------------------
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


def normalize_heading(title: str) -> str:
    """Entfernt Kapitel-/Abschnitts-Präfixe, Nummern und Deko."""
    if not title:
        return ""
    s = title.strip().lower()

   
    s = re.sub(r'^\s*(section|chapter|kapitel|abschnitt)\s*\d+[:.)-]*\s*', '', s, flags=re.I)

   
    s = re.sub(r'^\s*\d+(?:\.\d+)*\s*[:.)-]*\s*', '', s)

    
    s = s.rstrip(' :.-')
    s = re.sub(r'\s+', ' ', s)
    return s

# blacklist for heading 
HEADINGS_BLACKLIST_EQ = {
   
    "contents", "table of contents", "toc",
    "index", "references", "reference documentation",
    "company information", "company info",
    "revision history", "document history",
    "legal notice", "trademarks", "acknowledgements",
    "glossary", "contacts", "contact"
    }

    # # blacklist for heading  substring
HEADINGS_BLACKLIST_CONTAINS = {
    "reference documentation",
    "referenzdokumentation",
    "table of contents",
    "inhaltsverzeichnis",
    "revision history",
    "document history",
    "company information",
}
URL_RE = re.compile(r'https?://|www\.', re.I)

def title_matches_blacklist(title: str) -> bool:
    tnorm = normalize_heading(title)
    if tnorm in HEADINGS_BLACKLIST_EQ:
        return True
    return any(key in tnorm for key in HEADINGS_BLACKLIST_CONTAINS)


def url_ratio(text: str) -> float:
    if not text:
        return 0.0
    urls = len(URL_RE.findall(text))
    words = max(1, len(text.split()))
    return urls / words

def looks_like_link_table(text: str) -> bool:
   
    lines = text.splitlines()
    link_eq = sum(1 for ln in lines if "link" in ln.lower() and "=" in ln)
    pipes   = sum(1 for ln in lines if ln.count("|") >= 2)
    return link_eq >= 2 or pipes >= 6
def get_section_title_from_chunk(ch):
    hp = getattr(ch, "hierarchy_path", None)
    if isinstance(hp, list) and hp:
        last = hp[-1]
        if isinstance(last, dict):
            return last.get("title") or ""
    return ""

def first_line(text: str) -> str:
    return (text or "").split("\n", 1)[0].strip()

def should_drop_chunk(ch, ctx_text: str) -> bool:
    title_h = get_section_title_from_chunk(ch)
    title_f = first_line(ctx_text)

    # 1) Titel 
    if title_matches_blacklist(title_h) or title_matches_blacklist(title_f):
        return True

    # 2) Heuristik Links 
    if url_ratio(ctx_text) > 0.04 or looks_like_link_table(ctx_text):
        
        if any(k in normalize_heading(title_f) for k in ("reference", "referenz", "link", "documentation")) \
           or any(k in normalize_heading(title_h) for k in ("reference", "referenz", "link", "documentation")):
            return True

    return False
