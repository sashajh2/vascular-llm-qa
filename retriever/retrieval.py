import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load FAISS index and corpus (assumes pre-built)
faiss_index = faiss.read_index("retriever/faiss_index/my_index.faiss")
with open("retriever/faiss_index/corpus.json") as f:
    corpus = json.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Or your choice

def retrieve_docs(question, include_golden=True, top_k=4):
    # Embed question
    question_embedding = embedder.encode([question])
    _, indices = faiss_index.search(np.array(question_embedding), top_k)

    docs = [corpus[str(idx)] for idx in indices[0]]

    # Optionally add/remove golden doc
    if include_golden:
        golden_doc = get_golden_doc(question)  # Your function to fetch golden
        # Insert golden doc at the start (and remove a distractor if needed)
        docs = [golden_doc] + docs[:-1]

    return docs

def get_golden_doc(question):
    # Dummy golden doc fetcher for now
    return "Relevant information about the correct answer."


