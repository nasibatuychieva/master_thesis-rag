# Master Thesis: Retrieval-Augmented Generation with Knowledge Graphs

This repository contains the full implementation and evaluation code for my Master's thesis in Data Science.

The thesis investigates and compares classical Retrieval-Augmented Generation (RAG) pipelines with Knowledge Graph–enhanced RAG approaches for domain-specific technical question answering on product documentation.

---

## Research Objective

The goal of this thesis is to empirically evaluate whether integrating Knowledge Graphs (KGs) into Retrieval-Augmented Generation pipelines improves:

- Answer correctness
- Coverage of technical details
- Faithfulness to source documents
- Robustness for complex, multi-entity questions

The experiments are conducted on publicly available technical documentation (e.g., Arduino-style product manuals).

---

## Core Concepts

- **RAG (Retrieval-Augmented Generation)**
- **Knowledge Graph Construction & Retrieval**
- **Knowledge Graph Community Detection Construction**
- **Hybrid Retrieval ( Sparse + Dense + GraphRAG based)**
- **LLM-based Answer Generation**
- **LLM-as-a-Judge Evaluation**

---

## Repository Structure

```text
MASTER_THESIS-RAG/
│
├── main/
│   ├── chunking/               # Document chunking and preprocessing
│
│   ├── documents/              # Raw and processed Arduino documentation
│
│   ├── evaluation/             # Evaluation of RAG and RAG+KG approaches
│   │   ├── evaluation_datasets/  # Question–answer datasets with gold answers
│   │   ├── llm_as_judge/          # LLM-as-a-Judge evaluation framework
│   │   ├── judge_results/         # Aggregated evaluation scores
│   │   ├── log_results/           # Logged model responses and contexts
│   │   └── debug_traces/          # Debug and trace outputs
│
│   ├── knowledge_graph/        # Knowledge Graph construction and retrieval
│   │   ├── create_graph/         # Knowledge Graph creation
│   │   ├── build_communities/    # Community detection in the KG
│   │   └── retrieve_graph/       # KG-based and hybrid KG+RAG retrieval
│
│   ├── only_rag_pipelines/     # RAG pipelines without Knowledge Graph
│   │   ├── naive_rag/            # Baseline vector-based RAG
│   │   ├── create_rag/           # Retriever and index creation
│   │   └── advanced_rag/         # Enhanced RAG variants
│   
|   requirements.txt            # Python package dependencies
└── README.md                   # Project description and instructions


