from dotenv import load_dotenv
load_dotenv()

import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from main.evaluation.logger import log_antwort

# ---------------------------------------------------------------------------
# 1) LLM und Embeddings
# ---------------------------------------------------------------------------

llm = ChatLlamaCpp(
    model_path=r"C:\models\qwen2.5-7b-instruct-q3_k_m.gguf",
    temperature=0,
    n_ctx=4096,
    max_tokens=1024, 
    n_threads=4,
    n_gpu_layers=0,
)

embedding_provider = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# 2) Neo4j-Verbindung
# ---------------------------------------------------------------------------

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="testmaster123",
)


# ---------------------------------------------------------------------------
# 3) VectorStore mit Knowledge-Graph-Retrieval (RAG-Teil)
# ---------------------------------------------------------------------------
def expand_product(product_id: str):
    """
    Holt alle Knoten und Beziehungen in einem 4-Hop-Umfeld um ein Produkt,
    ohne die Beziehungstypen explizit zu kennen.
    """
    query = """
    MATCH (p:Product {id: $pid})
    MATCH path = (p)-[*1..4]-(x)
    RETURN DISTINCT x, path
    """
    return graph.query(query, {"pid": product_id})

retrieval_query = """
// 1) Produkt zum Chunk
MATCH (node)-[:PART_OF]->(p:Product)

// 2) Dokument
OPTIONAL MATCH (node)-[:IN_DOCUMENT]->(d:Document)

// 3) Tutorial
OPTIONAL MATCH (node)-[:IN_TUTORIAL]->(t:Tutorial)

// 4) Elements des Produkts
OPTIONAL MATCH (p)-[rel]->(el:Element)
WITH node, score, p, d,
     coalesce(t.id, node.tutorial) AS tutorial_id,
     collect(DISTINCT {element: el.id, rel_type: type(rel)}) AS elements

// 5) Entities
OPTIONAL MATCH (node)-[:HAS_ENTITY]->(e1)
OPTIONAL MATCH (e1)-[r]-(e2)
WHERE (node)-[:HAS_ENTITY]->(e2)

WITH node, score, p, d, tutorial_id, elements,
     collect(DISTINCT e1.id) AS ents,
     collect(DISTINCT apoc.text.join([
         labels(startNode(r))[0], coalesce(startNode(r).id,''),
         type(r),
         labels(endNode(r))[0], coalesce(endNode(r).id,'')
     ], ' ')) AS kg

RETURN
  node.text AS text,
  score,
  {
    chunk:   node.id,
    product: p.id,
    document: d.id,
    elements: elements,   // <--- JETZT wirklich drin
    entities: ents,
    kg: kg,
    tutorial: tutorial_id
  } AS metadata;



"""

# VectorStore aus EXISTIERENDEM Vector-Index "chunkVector"
chunk_vector = Neo4jVector.from_existing_index(
    embedding=embedding_provider,
    graph=graph,
    index_name="chunkVector",              # Name des Vector-Index in Neo4j
    embedding_node_property="textEmbedding",
    text_node_property="text",
    node_label="Chunk",
    retrieval_query=retrieval_query,       # hier kommt der KG-Teil rein
)

retriever = chunk_vector.as_retriever(
    search_kwargs={"k": 4}
)


# ---------------------------------------------------------------------------
# 4) Prompt + Kontext-Formatierung (Text + KG)
# ---------------------------------------------------------------------------

instructions = (
    "Use the given context to answer the question. "
    "The context comes from chunks of product documentation and a knowledge graph. "
    "Reply with an answer that includes the product id and chunk id from metadata, "
    "and mention relevant entities if helpful. "
    "If you don't know, say you don't know."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    parts = []
    for d in docs:
        md = d.metadata or {}
        chunk_id = md.get("chunk")
        product_id = md.get("product")
        document_name = md.get("document")     # <--- NEU
        entities = md.get("entities") or []
        kg_triples = md.get("kg") or []
        tutorial_name = md.get("tutorial") or []
        elements = md.get("elements") or []

        part = [
            f"Chunk ID: {chunk_id}",
            f"Product ID: {product_id}",
            f"Document: {document_name}",       # <--- NEU
            f"Entities: {entities}",
            f"KG triples: {kg_triples}",
            f"Tutorial: {tutorial_name}",
            f"Elements: {elements}",
            "",
            f"Text:\n{d.page_content}",
        ]
        parts.append("\n".join(part))

    if not parts:
        return "NO CONTEXT FOUND."
    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(parts)



# ---------------------------------------------------------------------------
# 5) RAG-Chain (LCEL): Frage -> Retriever (Vektor + KG) -> Prompt -> LLM
# ---------------------------------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,   # nutzt Embeddings + KG-Query
        "input": RunnablePassthrough(),       # Frage unverändert weiterreichen
    }
    | prompt
    | llm
    | StrOutputParser()                       # ChatMessage -> String
)


# ---------------------------------------------------------------------------
# 6) Dynamische Erkennung: Graph-Fragen (Produkte nach Kategorie/Familie)
# ---------------------------------------------------------------------------

def extract_category_from_question(question: str) -> str | None:
    """
    Versucht aus der Frage eine Kategorie/Familie herauszuziehen.
    Funktioniert für einfache englische & deutsche Fragen wie:
      - "Which products belong to Modulino Family?"
      - "Which products belong to the Modulino family?"
      - "Welche Produkte gehören zur Modulino Familie?"
    Gibt den gefundenen String zurück (z.B. 'Modulino Family'), sonst None.
    """
    q = question.strip().rstrip("?.!").lower()

    # Englische Varianten
    patterns_en = [
        r"which products(?:\s+do)?\s+belong to\s+(.*)",
        r"what products(?:\s+do)?\s+belong to\s+(.*)",
        r"which products are in\s+(.*)",
        r"list (?:all )?products in\s+(.*)",
    ]

    # Deutsche Varianten
    patterns_de = [
        r"welche produkte\s+gehören\s+zur\s+(.*)",
        r"welche produkte\s+gehören\s+zu\s+(.*)",
        r"welche produkte\s+gibt es in\s+(.*)",
    ]

    for pat in patterns_en + patterns_de:
        m = re.match(pat, q)
        if m:
            cat = m.group(1).strip()
            # evtl. Wörter wie "family", "familie", "category", "kategorie" am Ende kürzen
            cat = re.sub(r"\b(family|familie|category|kategorie)\b$", "", cat).strip()
            return cat if cat else None

    return None
def extract_product_from_question(question: str):
    # einfache heuristik
    possible_products = graph.query("MATCH (p:Product) RETURN p.id AS id")
    ids = [row["id"].lower() for row in possible_products]

    q = question.lower()
    for pid in ids:
        if pid.lower() in q:
            return pid
    return None 


def answer_products_by_category(category_query: str) -> str:
    """
    Nutzt den Knowledge Graph, um ALLE Produkte zu einer Kategorie/Familie zu holen.
    Sucht case-insensitive in pc.id.
    """
    records = graph.query(
        """
        MATCH (pc:ProductCategory)
        WHERE toLower(pc.id) CONTAINS toLower($cat)
        MATCH (pc)-[:HAS_PRODUCT]->(p:Product)
        RETURN pc.id AS y, collect(DISTINCT p.id) AS products
        """,
        params={"cat": category_query},
    )

    if not records:
        return (
            f"I couldn't find any product category matching '{category_query}'. "
            f"Maybe the name is different in the documentation/graph."
        )

    lines = []
    for row in records:
        cat = row["category"]
        products = row["products"] or []
        if not products:
            lines.append(f"Category '{cat}' has no products in the graph.")
        else:
            header = f"Category '{cat}' has {len(products)} product(s):"
            prod_lines = "\n".join(f"- {p}" for p in products)
            lines.append(header + "\n" + prod_lines)

    return "\n\n".join(lines)


def answer_question(question: str) -> str:
    """
    Dynamische Entscheidung:
      - Wenn Frage nach 'Welche Produkte gehören zu Kategorie/Familie X?' aussieht
        → direkte Graph-Abfrage (Cypher, global, ALLE Produkte).
      - Sonst → normaler RAG+KG-Flow (Vectorindex + Kontext + LLM).
    """
    category = extract_category_from_question(question)
    if category:
        # → Knowledge-Graph direkt nutzen (ALLE Produkte)
        return answer_products_by_category(category)

    # → Standard: RAG+KG
    result = rag_chain.invoke(question)

    # Dokumentnamen sammeln
    docs = chunk_vector.similarity_search(question, k=4)

    doc_names = {d.metadata.get("document") for d in docs if d.metadata}


    return result + "\n\nSources:\n" + "\n".join(f"- {n}" for n in doc_names)



# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("RAG + Knowledge Graph Chat. Enter 'exit' for finishing.\n")

    SCRIPT_NAME = "retriever_new.py"  # oder automatisch: os.path.basename(__file__)

    while (q := input("> ")).strip().lower() != "exit":
        if not q:
            continue

        out = answer_question(q)
        print("\n" + out + "\n")

       # log the answer
        log_antwort(SCRIPT_NAME, q, out)

