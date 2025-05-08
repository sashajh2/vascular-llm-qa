import argparse
from openai import OpenAI
import os
import pandas as pd
from retriever.retrieval import retrieve_docs
from utils.formatting import format_mcq_test_example
from utils.data_utils import load_dataset_from_jsonl

def generate_mcq_test_dataset(dataset):
    """Generate golden and distractor examples for each question."""
    prompts= []

    for example in dataset:
        # Golden + distractors version
        context_chunks_relevant = retrieve_docs(example["question"], include_relevant=True)
        prompt_relevant = format_mcq_test_example(example, context_chunks_relevant)
        prompts.append(prompt_relevant)
    return prompts

def generate_response(prompt, i, client):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering multiple choice questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200  # Adjust based on output needs
        )
        # Corrected: Extract response text
        output = completion.choices[0].message.content.strip()
        return output
    except Exception as e:
        print(f"Error generating response for prompt {i}: {e}")
        return []

def run_gpt_and_save(test_data_path, output_path, client):
    test_data = load_dataset_from_jsonl(test_data_path)
    prompts = generate_mcq_test_dataset(test_data)
    results = []
    for i, example in enumerate(test_data):
        prompt = prompts[i]
        response = generate_response(prompt, i, client)
        
        results.append({
            "id": example['id'],
            "question": example["question"],
            "choices": example["choices"],
            "correct_answer": example["correct_answer"],
            "model_response": response
        })
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(results)} predictions to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    args = parser.parse_args()

    api_key = args.api_key
    client = OpenAI(api_key=api_key)

    run_gpt_and_save(args.data_path, args.output_path, client)