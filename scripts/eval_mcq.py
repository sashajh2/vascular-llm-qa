import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever.retrieval import retrieve_docs
from utils.formatting import format_mcq_example
from tqdm import tqdm

def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)[0]
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)

def main(model_path, data_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).eval().to("cuda")

    data = load_dataset(data_path)
    outputs = []

    for example in tqdm(data):
        context_chunks = retrieve_docs(example["question"], include_relevant=True)
        prompt, _ = format_mcq_example(example, context_chunks, include_relevant=True)

        generation = generate_response(model, tokenizer, prompt)
        outputs.append({
            "question": example["question"],
            "gold_answer": example["answer"],
            "model_response": generation,
            "prompt_used": prompt
        })

    with open(output_path, "w") as f:
        for entry in outputs:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.output_path)
