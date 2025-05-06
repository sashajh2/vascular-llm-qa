# This doc: 
# 1. Load documents from corpus/
# 2: Embed them
# 3: Store vectors and metadata in FAISS index
# 4: Save metadata to JSON for reverse lookup (if needed)

### General RAG Pipeline
# Step 1: Document preprocessing
#   Put raw docs in data/raw
#   Write a function called "extract_text(doc)" in data_utils
#       Preprocess extract text using pdfplumber/docx/etc
#       Clean (remove headers/footers/artifacts)
#       Call file-type specific helper functions from this overarching extract_text func
#   Write a function called "chunk_text_with_tokenizer(text, tokenizer chunk_size, overlap)" to chunk the text
#   Example below. Any tokenizer is fine. Just make sure it matches the embedding for later
#   Otherwise, you can chunk by words
#   
#   Save the chunks to txt files in the corpus

# Step 2: Build FAISS Index (build_index.py)
#       1. Load chunks from retriever/corpus
#       2. Embed the chunks using the same embedding model as the tokenizer
#       3. Store the vectors in the FAISS index
#       4. Save metadata (chunk_to_file mapping) as a JSON

import os
# from utils.data_utils import extract_text  # Your helper for text extraction
from transformers import AutoTokenizer

def extract_text(path):
    with open(path) as f:
        return f.read()


def chunk_text_with_tokenizer(text, tokenizer, chunk_size=500, overlap=100):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

#This should be put in a util file (maybe data_utils). Call via the cmd line
def process_documents(raw_dir="data/raw/rag", corpus_dir="retriever/corpus", chunk_size=500, overlap=100, tokenizer_name="FreedomIntelligence/Apollo-0.5B"):
    """
    Processes raw documents: extracts text, chunks it, and saves to corpus directory.
    
    Args:
        raw_dir (str): Directory containing raw documents (PDF, DOCX, TXT).
        corpus_dir (str): Directory to save processed text chunks.
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
        tokenizer_name (str): HuggingFace tokenizer to use for chunking.
    """
    os.makedirs(corpus_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for doc_name in os.listdir(raw_dir):
        doc_path = os.path.join(raw_dir, doc_name)
        try:
            text = extract_text(doc_path)
            chunks = chunk_text_with_tokenizer(text, tokenizer, chunk_size, overlap)
            
            # Save chunks directly
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{os.path.splitext(doc_name)[0]}_chunk{i}.txt"
                chunk_path = os.path.join(corpus_dir, chunk_filename)
                with open(chunk_path, "w") as f:
                    f.write(chunk)
            print(f"Processed {doc_name}: {len(chunks)} chunks saved.")
        except Exception as e:
            print(f"Error processing {doc_name}: {e}")

### Pseudocode for build_index.py
from sentence_transformers import SentenceTransformer
import faiss
import os
import json

process_documents( 
    raw_dir="data/raw/rag",
    corpus_dir="retriever/corpus_100",
    chunk_size=100,
    overlap=20,
    tokenizer_name="FreedomIntelligence/Apollo-0.5B"
)

# 1. Load text chunks
corpus_dir = "retriever/corpus_100/"
documents = []
metadata = []

for file in os.listdir(corpus_dir):
    with open(os.path.join(corpus_dir, file)) as f:
        text = f.read()
        documents.append(text)
        metadata.append({"filename": file})

# 2. Embed the documents
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# 3. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Save index and metadata
faiss.write_index(index, "retriever/faiss_index/corpus_100.index")

with open("retriever/faiss_index/metadata_100.json", "w") as f:
    json.dump(metadata, f)