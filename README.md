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
- **Hybrid Retrieval (BM25 + Dense + Graph-based)**
- **LLM-based Answer Generation**
- **LLM-as-a-Judge Evaluation**
- **Community Detection in Knowledge Graphs**

---

## Repository Structure

```text
MASTER_THESIS-RAG/
│
├── .venv_Python11_New/          # Python virtual environment

│
├── main/
│   ├── chunking/               # Document chunking and preprocessing
│   ├── documents/              # Raw and processed input documents
│   ├── evaluation/             # Evaluation pipelines, datasets, judges
│   ├── knowledge_graph/        # Knowledge Graph construction, Community Detection & KG retrieval
│   ├── only_rag_pipelines/     # Baseline RAG pipelines (without KG)
│   ├── out_aktuell/            # Chunking  outputs (latest runs)
│   ├── out_aktuell_processed_files/  # Processed experiment results
│   ├── ressources/             # Auxiliary resources (configs, helpers)
│  
│

├── docker.txt                  # Notes / instructions for Docker usage
├── .env                        # Environment variables (not committed)
├── .gitignore
│
├── pyproject.toml              # Project configuration
├── requirements-retriever.txt  # Python dependencies
└── README.md                   # This file
