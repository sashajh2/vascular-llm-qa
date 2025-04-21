# For data conversion helper functions
import pandas as pd
import json

def convert_xlsx_to_jsonl(xlsx_path, output_path):
    df = pd.read_excel(xlsx_path)

    examples = []
    for _, row in df.iterrows():
        example = {
            "question": row["question"],
            "choices": [row.get(f"{c}_choice", "") for c in "ABCD"],
            "correct_answer": row["correct_answer"],
            "explanation": row.get("explanation", ""),
            "source": row.get("source", "")
        }
        examples.append(example)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

