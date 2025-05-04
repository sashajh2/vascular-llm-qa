import argparse
import json
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever.retrieval import retrieve_docs
from utils.formatting import format_mcq_test_example
from utils.data_utils import load_dataset_from_jsonl


def generate_mcq_test_dataset(dataset, tokenizer):
    """Generate golden and distractor examples for each question."""
    prompts= []

    for example in dataset:
        # Golden + distractors version
        context_chunks_relevant = retrieve_docs(example["question"], include_relevant=True)
        prompt_relevant = format_mcq_test_example(example, context_chunks_relevant)
        prompts.append(prompt_relevant)
    return prompts


def run_inference_and_save(model_name_or_path, test_data_path, output_path):
    model_name_or_path = os.path.abspath(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    test_data = load_dataset_from_jsonl(test_data_path)
    prompts = generate_mcq_test_dataset(test_data, tokenizer)
    results = []
    for i, example in enumerate(test_data):
        prompt = prompts[i]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.replace(prompt, "").strip()
        
        results.append({
            "id": example['id'],
            "question": example["question"],
            "choices": example["choices"],
            "correct_answer": example["correct_answer"],
            "model_response": response
        })
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    run_inference_and_save(args.model_path, args.data_path, args.output_path)
