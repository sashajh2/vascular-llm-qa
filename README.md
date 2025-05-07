# 🧠 Vascular LLM QA

This repository contains the code and data for a project evaluating the effectiveness of large language models (LLMs) on vascular medicine multiple-choice question answering (MCQA). We evaluate baseline model performance, apply retrieval-augmented generation (RAG), and implement retrieval-augmented fine-tuning (RAFT).

---

## 📁 Repository Structure


---

## 🧾 Data Overview

- `data/raw/`: Raw multiple-choice question sets (e.g. `MCQ2.0.xlsx`) and original source materials.
- `data/rag/`: Corpus used for retrieval, prior to chunking or indexing.
- `data/processed/`: Cleaned and preprocessed JSONL files. Includes:
  - `mcq.jsonl`: Original cleaned MCQ data.
  - `mcq_train.jsonl`, `mcq_test.jsonl`: Training and testing splits.
  - `test_prompts/`: Prompt-structured MCQs used during inference or evaluation.

---

## 📓 Notebook Breakdown

- `notebooks/base/`: Evaluates model performance with no retrieval or fine-tuning.
- `notebooks/rag/`: Evaluates RAG-enhanced models using FAISS-based retrieval.
- `notebooks/RAFT/`: Fine-tunes models with RAFT and evaluates them.
  - **Note**: Due to GPU memory constraints, we recommend training and evaluating in a single Colab session without saving intermediate checkpoints. RAFT models with >3B parameters (like LLaMA-2 7B or Phi-2) may not run on standard Colab GPUs. Only Apollo (0.5B) and JSL-Medphi (2.7B) are reliably trainable on 40GB Colab instances.

---

## 🔍 Retriever

- Retrieval logic is located in the `retriever/` directory.
- Two chunking strategies:
  - `corpus/`: 500-token chunks with 100-token overlap.
  - `corpus_100/`: 100-token chunks with 20-token overlap.
- Use `build_index.py` or `build_index_100.py` to build a FAISS index.
- Retrieval is handled by `retrieval.py` and `retrieval_100.py`.

---

## ⚙️ Scripts

- `scripts/fine_tune_mcq.py`: Trains a causal language model using the RAFT method (includes tokenizer, LoRA support).
- `scripts/run_inference.py`: Loads model and generates responses in multiple-choice format with optional RAG context.

---

## 🧪 Tests & Utils

- `tests/`: Sanity checks for retrieval and synthetic test question generation.
- `utils/`: Common helper functions for formatting, metric computation, and data conversion.

---

## 📦 Installation

We recommend using a Python 3.10+ virtual environment.

```bash
git clone https://github.com/sashajh2/vascular-llm-qa.git
cd vascular-llm-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu
