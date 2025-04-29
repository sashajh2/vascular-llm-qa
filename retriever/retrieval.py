import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import random

# Load FAISS index and corpus (assumes pre-built)
faiss_index = faiss.read_index("retriever/faiss_index/corpus.index")
with open("retriever/faiss_index/metadata.json") as f:
    metadata = json.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_chunk_by_index(idx, chunk_dir="retriever/corpus"):
    """Given a FAISS index, load the corresponding chunk text."""
    filename = metadata[idx]["filename"]
    path = os.path.join(chunk_dir, filename)
    with open(path, "r") as f:
        return filename, f.read()

# next; include golden for training
def retrieve_docs(question, include_relevant=True, top_k=4):
    """Retrieve documents for a question.
    
    If include_relevant=True: return top-k FAISS retrieved docs.
    If include_relevant=False: return true distractors, sampled randomly while excluding top matches.
    """
    
    # Embed the question
    question_embedding = embedder.encode([question])
    question_embedding = np.array(question_embedding).reshape(1, -1)
    
    # Search FAISS index
    _, indices = faiss_index.search(question_embedding, top_k)
    retrieved_indices = list(indices[0])

    if include_relevant:
        # Return the top-k most similar documents
        return [get_chunk_by_index(idx) for idx in retrieved_indices[:top_k]]
    
    else:
        # Build candidate pool by excluding top-k retrieved
        all_indices = set(range(len(metadata)))
        blocked_indices = set(retrieved_indices[:top_k])
        candidate_indices = list(all_indices - blocked_indices)
        
        # Randomly sample true distractors
        distractor_indices = random.sample(candidate_indices, top_k)
        return [get_chunk_by_index(idx) for idx in distractor_indices]
        


