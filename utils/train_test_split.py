import argparse
import json
import random
import os

# Assume linked pairs are already defined:
linked_pairs = [
    ("1a", "1b"), ("2a", "2b"), ("3a", "3aa"), ("3b", "3ba"),
    ("3c", "3ca"), ("4a", "4b"), ("5a", "5b"), ("6a", "6b"),
    ("7a", "7b"), ("8a", "8b"), ("9a", "9b"), ("10a", "10b"), ("11a", "11b")
]
linked_ids = set(q for pair in linked_pairs for q in pair)

def has_explanation(example):
    return example.get('explanation', "").strip() != ""

def create_stratify_label(example):
    return f"{example['source']}_{'has_exp' if has_explanation(example) else 'no_exp'}"

def train_test_split_jsonl(input_path, train_output_path, test_output_path, test_size=0.2, seed=42):
    # Load all examples from input JSONL
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Add stratify_label
    for ex in examples:
        ex['stratify_label'] = create_stratify_label(ex)

    # Group by stratify_label
    stratify_groups = {}
    for ex in examples:
        label = ex['stratify_label']
        if label not in stratify_groups:
            stratify_groups[label] = []
        stratify_groups[label].append(ex)

    train_set, test_set = [], []

    random.seed(seed)

    for label, group in stratify_groups.items():
        # Separate linked and unlinked examples
        linked = [ex for ex in group if ex['id'] in linked_ids]
        unlinked = [ex for ex in group if ex['id'] not in linked_ids]

        # Build linked pairs properly
        id_to_ex = {ex['id']: ex for ex in linked}
        linked_pairs_present = []
        for pair in linked_pairs:
            if pair[0] in id_to_ex and pair[1] in id_to_ex:
                linked_pairs_present.append((id_to_ex[pair[0]], id_to_ex[pair[1]]))

        random.shuffle(linked_pairs_present)
        n_test_linked = int(len(linked_pairs_present) * test_size)

        test_linked = [item for pair in linked_pairs_present[:n_test_linked] for item in pair]
        train_linked = [item for pair in linked_pairs_present[n_test_linked:] for item in pair]

        # Split unlinked normally
        random.shuffle(unlinked)
        n_test_unlinked = int(len(unlinked) * test_size)
        test_unlinked = unlinked[:n_test_unlinked]
        train_unlinked = unlinked[n_test_unlinked:]

        # Combine splits
        train_set.extend(train_linked + train_unlinked)
        test_set.extend(test_linked + test_unlinked)

    # Write train and test sets to new JSONL files
    with open(train_output_path, 'w') as f:
        for ex in train_set:
            del ex['stratify_label']  # Remove internal stratify field
            f.write(json.dumps(ex) + "\n")

    with open(test_output_path, 'w') as f:
        for ex in test_set:
            del ex['stratify_label']
            f.write(json.dumps(ex) + "\n")

    print(f"Train size: {len(train_set)} examples")
    print(f"Test size: {len(test_set)} examples")

def main(input_path, train_output_path, test_output_path):
    print(f"Splitting {input_path} into:")
    print(f"   ➤ {train_output_path} (train)")
    print(f"   ➤ {test_output_path} (test)")
    train_test_split_jsonl(input_path, train_output_path, test_output_path)
    print("✅ Split complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL dataset into stratified train/test sets.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--train_output", type=str, required=True, help="Path to output train JSONL file.")
    parser.add_argument("--test_output", type=str, required=True, help="Path to output test JSONL file.")
    args = parser.parse_args()

    main(args.input, args.train_output, args.test_output)