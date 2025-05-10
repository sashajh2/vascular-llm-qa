import json

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def build_explanation_lookup(gpt_exp_path):
    gpt_data = load_jsonl(gpt_exp_path)
    return {example["id"]: example["explanation"] for example in gpt_data if "explanation" in example}

def add_explanations(split_path, output_path, explanation_dict):
    data = load_jsonl(split_path)
    for example in data:
        example_id = example.get("id")
        if example_id in explanation_dict:
            example["explanation"] = explanation_dict[example_id]
    save_jsonl(data, output_path)

def main():
    # File paths
    gpt_exp_path = "data/processed/mcq_gpt_exp.jsonl"
    test_path = "data/processed/mcq_test.jsonl"
    train_path = "data/processed/mcq_train.jsonl"
    test_output = "data/processed/mcq_test_gpt.jsonl"
    train_output = "data/processed/mcq_train_gpt.jsonl"

    explanation_dict = build_explanation_lookup(gpt_exp_path)

    add_explanations(test_path, test_output, explanation_dict)
    add_explanations(train_path, train_output, explanation_dict)

    print("Finished writing augmented splits with GPT explanations.")

if __name__ == "__main__":
    main()
