import re
from typing import List

# -----------------------
# 0)Helpers
# -----------------------
ABBREV = r"(?:e\.g|i\.e|etc|Mr|Mrs|Dr|vs)\."
UNIT_PATTERNS = [
    (r"(?P<num>\d)\s*(?P<u>MHz|kHz|Hz|GHz|MB|GB|kB|KB|V|mV|A|mA|°C)", r"\g<num> \g<u>"),
    (r"(?P<num>\d)\s*(?P<u>mm|cm)", r"\g<num> \g<u>")
]

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # NBSP
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()

def normalize_units(text: str) -> str:
    for pat, rep in UNIT_PATTERNS:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    # Simplify ranges w
    text = re.sub(r"(-?\d+)\s*°C\s*[-–]\s*(-?\d+)\s*°C", r"\1 °C to \2 °C", text)
    text = re.sub(r"(\d+)\s*-\s*(\d+)\s*V", r"\1 V to \2 V", text)
    return text

def join_wrapped_lines(text: str) -> str:

    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            out.append("") ; continue

        if (not re.match(r"^\s*[-•\u2022]|^\s*[A-Z][A-Za-z0-9 \-/]+\:$", ln)
            and not re.search(r"[.!?]$", ln)
            and i+1 < len(lines) and not re.match(r"^\s*[-•\u2022]|^\s*[A-Z][A-Za-z0-9 \-/]+\:$", lines[i+1])):
           
            if out and out[-1]:
                out[-1] += " " + ln.strip()
            else:
                out.append(ln.strip())
        else:
            out.append(ln.strip())
    return "\n".join(out)

# -----------------------
# 1) Key:Value / Table-> Sentences
# -----------------------
def kv_to_sentence(key: str, value: str, subject: str) -> str:
    k = key.lower().strip()
    v = value.strip().rstrip(".")
    if not v:
        return ""

    # Domainspecific templates
    if "microcontroller" in k:
        return f"The {subject} uses a {v} microcontroller."
    if "wireless" in k or "connectivity" in k or "bluetooth" in k or "wi-fi" in k:
        return f"The {subject} supports {v} connectivity."
    if "ethernet" in k:
        return f"The {subject} includes an Ethernet transceiver ({v})."
    if "security" in k or "secure" in k:
        return f"The {subject} includes the secure element {v}."
    if "external memory" in k or "memory" in k:
        return f"The {subject} provides external memory: {v}."
    if "power" in k or "power supply" in k:
        return f"The {subject} can be powered via {v}."
    if "dimensions" in k:
        return f"The {subject} measures {v}."
    if "peripherals" in k or "interfaces" in k:
        return f"The {subject} supports the following interfaces: {v}."
    
    return f"The {subject} has {key.strip()}: {v}."

def transform_key_value_lines(text: str, subject: str) -> str:
    out: List[str] = []
    for ln in text.splitlines():
        m_colon = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /+\-\.()]+)\s*:\s*(.+)$", ln)
        m_table = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /+\-\.()]+)\s{2,}(.+)$", ln)
        if m_colon:
            s = kv_to_sentence(m_colon.group(1), m_colon.group(2), subject)
            out.append(s)
        elif m_table:
            s = kv_to_sentence(m_table.group(1), m_table.group(2), subject)
            out.append(s)
        else:
            out.append(ln)
    return "\n".join([x for x in out if x])

# -----------------------
# 2) Bullet-Points 
# -----------------------
def transform_bullets(text: str, subject: str) -> str:
    out = []
    for ln in text.splitlines():
        if re.match(r"^\s*[-•\u2022]\s*", ln):
            item = re.sub(r"^\s*[-•\u2022]\s*", "", ln).strip()
            # If at the beginning alread verb, then leave, else "has"/"supports"
            if re.match(r"^(has|supports|includes|uses|provides|offers|contains)\b", item, flags=re.I):
                out.append(f"The {subject} {item[0].lower() + item[1:]}")
            else:
                # Heuristik: Interfaces/Features
                if re.search(r"\b(I2C|SPI|UART|CAN|PWM|ADC|DAC|Wi-?Fi|Bluetooth)\b", item, flags=re.I):
                    out.append(f"The {subject} supports {item}.")
                else:
                    out.append(f"The {subject} has {item}.")
        else:
            out.append(ln)
    return "\n".join(out)

# -----------------------
# 3) Simple sentence building
# -----------------------
def ensure_sentence_punctuation(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if ln and not re.search(rf"{ABBREV}$|[.!?]$", ln):
            out.append(ln + ".")
        else:
            out.append(ln)
    return "\n".join(out)

# -----------------------
# 4)Sentence Splitter
# -----------------------
def split_sentences(text: str) -> List[str]:
    
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<![A-Z][a-z])(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in parts if p.strip()]

# -----------------------
# Master-Function
# -----------------------
def prepare_for_rebel(text: str, subject: str = "Portenta C33", return_sentences: bool = True):
    t = normalize_whitespace(text)
    t = join_wrapped_lines(t)
    t = transform_key_value_lines(t, subject)
    t = transform_bullets(t, subject)
    t = normalize_units(t)
    t = ensure_sentence_punctuation(t)
    return split_sentences(t) if return_sentences else t
