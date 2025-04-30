import json
from retriever.retrieval import retrieve_docs
from utils.formatting import format_mcq_example
import os

# === 1. Load mcq_test.jsonl ===
with open("data/processed/mcq_test.jsonl") as f:
    test_data = [json.loads(line) for line in f]

# === 2. Pick one example ===
example = test_data[47]  # or pick by index, etc.

# === 3. Retrieve relevant context chunks ===
relevant_context_chunks = retrieve_docs(example["question"], include_relevant=True, top_k=4)
irrelevant_context_chunks = retrieve_docs(example["question"], include_relevant=False, top_k=4)
# === 4. Format prompt ===
relevant_prompt, _ = format_mcq_example(example, relevant_context_chunks, include_relevant=True)
irrelevant_prompt, _ = format_mcq_example(example, irrelevant_context_chunks, include_relevant=True)

# === 5. Save to .txt file ===
relevant_output_path = "relevant_formatted_prompt_47.txt"
with open(relevant_output_path, "w") as f:
    f.write(relevant_prompt)

print(f"Relevant Prompt saved to: {relevant_output_path}")

irrelevant_output_path = "irrelevant_formatted_prompt_47.txt"
with open(irrelevant_output_path, "w") as f:
    f.write(irrelevant_prompt)

print(f"Irrelevant Prompt saved to: {irrelevant_output_path}")




