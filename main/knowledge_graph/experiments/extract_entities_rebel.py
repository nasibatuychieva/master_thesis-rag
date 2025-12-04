from transformers import pipeline
from tqdm import tqdm
import json

# Globale Variable für Lazy Loading
triplet_extractor = None

def load_pipeline():
    global triplet_extractor
    if triplet_extractor is None:
        print("📦 Modell-Pipeline will be loaded...")
        triplet_extractor = pipeline(
            'text2text-generation',
            model='Babelscape/rebel-large',
            tokenizer='Babelscape/rebel-large'
        )
        print("Modell-Pipeline is loaded.")

def parse_triplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation:
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation:
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject and relation and object_:
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})

    return triplets

def extract_relations(text):
    load_pipeline()
    response = triplet_extractor(
        text,
        return_tensors=True,
        return_text=False
    )
    extracted_text = triplet_extractor.tokenizer.batch_decode([response[0]["generated_token_ids"]])
    return parse_triplets(extracted_text[0])

# -------------------- Mainprogramm --------------------

def process_extractions (input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_triples = []
    print(f"Started Extraktion for {len(lines)} Chunks...")

    for line in tqdm(lines, desc="🔍 Process Chunks"):
        data = json.loads(line)
        text = data.get("text", "")
        if text.strip():
            triples = extract_relations(text)
            all_triples.extend(triples)

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(all_triples, out_f, indent=2, ensure_ascii=False)

    print(f"Finished the extraction for {len(all_triples)} and stored it in {output_path}.")
