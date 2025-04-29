import pandas as pd
import json
import re
from datasets import Dataset, concatenate_datasets
from random import random

def extract_answer_choices(question_text):
    pattern = r"\s+([A-Z])\.\s+((?:.|\n)*?)(?=\s+[A-Z]\.|$)"
    matches = re.findall(pattern, question_text)

    if not matches:
        raise ValueError("No answer choices found in question.")

    # Extract choice text (without label)
    answer_choices = [choice.strip() for _, choice in matches]

    # Get index where first labeled answer choice starts to extract question stem
    match_start = re.search(pattern, question_text).start()
    question_stem = question_text[:match_start].strip()

    return question_stem, answer_choices

def convert_xlsx_to_jsonl(xlsx_path, output_path):
    df = pd.read_excel(xlsx_path)

    examples = []
    for _, row in df.iterrows():
        full_question_text = row["question"]
        try: 
            question_stem, answer_choices = extract_answer_choices(full_question_text)
        except Exception as e:
            print(f"Error parsing choices for question: {full_question_text}")
            raise e
        
        question_id = str(row['question_id'])
        raw_answers = str(row['answer'])
        answer_letters = re.split(r"[,\s]+", raw_answers.strip())

        explanation = row.get("explanation", "")
        source = row.get("source", "")

        # FIX: Do not inject placeholders here
        if pd.isna(explanation):
            explanation = ""
        if pd.isna(source):
            source = ""
    
        example = {
            "id": question_id,
            "question": question_stem,
            "choices": answer_choices,
            "correct_answer": answer_letters,
            "explanation": explanation,
            "source": source
        }
        examples.append(example)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def has_explanation(example):
    """Helper to check if explanation exists."""
    return example.get('explanation', "").strip() != ""

linked_pairs = [
    ("1a", "1b"), ("2a", "2b"), ("3a", "3aa"), ("3b", "3bb"),
    ("3c", "3cc"), ("4a", "4b"), ("5a", "5b"), ("6a", "6b"),
    ("7a", "7b"), ("8a", "8b"), ("9a", "9b"), ("10a", "10b"), ("11a", "11b")
]
linked_ids = set(q for pair in linked_pairs for q in pair)

def train_test_split(dataset: Dataset, test_size=0.2, seed=42):
    def create_stratify_label(example):
        return f"{example['source']}_{'has_exp' if has_explanation(example) else 'no_exp'}"

    dataset = dataset.map(lambda x: {"stratify_label": create_stratify_label(x)})
    unique_labels = list(set(dataset['stratify_label']))
    train_splits, test_splits = [], []

    for label in unique_labels:
        group = dataset.filter(lambda x: x['stratify_label'] == label)
        group = group.shuffle(seed=seed)

        # Split into linked/unlinked based on defined pairs
        linked = group.filter(lambda x: x['id'] in linked_ids)
        unlinked = group.filter(lambda x: x['id'] not in linked_ids)

        # Build a mapping from id → row
        id_to_example = {row['id']: row for row in linked}

        # Filter and group linked pairs that are actually present
        linked_pairs_selected = []
        for pair in linked_pairs:
            if pair[0] in id_to_example and pair[1] in id_to_example:
                linked_pairs_selected.append((id_to_example[pair[0]], id_to_example[pair[1]]))

        random.seed(seed)
        random.shuffle(linked_pairs_selected)

        n_test_linked = int(len(linked_pairs_selected) * test_size)

        test_linked_examples = [item for pair in linked_pairs_selected[:n_test_linked] for item in pair]
        train_linked_examples = [item for pair in linked_pairs_selected[n_test_linked:] for item in pair]

        test_linked_ds = Dataset.from_list(test_linked_examples) if test_linked_examples else Dataset.from_dict({})
        train_linked_ds = Dataset.from_list(train_linked_examples) if train_linked_examples else Dataset.from_dict({})

        # Split unlinked as usual
        n_test_unlinked = int(len(unlinked) * test_size)
        test_unlinked = unlinked.select(range(n_test_unlinked))
        train_unlinked = unlinked.select(range(n_test_unlinked, len(unlinked)))

        # Combine everything
        train_group = concatenate_datasets([train_unlinked, train_linked_ds]) if len(train_linked_ds) > 0 else train_unlinked
        test_group = concatenate_datasets([test_unlinked, test_linked_ds]) if len(test_linked_ds) > 0 else test_unlinked

        train_splits.append(train_group)
        test_splits.append(test_group)

    train_dataset = concatenate_datasets(train_splits)
    test_dataset = concatenate_datasets(test_splits)

    return train_dataset, test_dataset
