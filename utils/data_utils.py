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

def load_dataset_from_jsonl(jsonl_path):
    """Load and parse a JSONL dataset for instruction fine-tuning."""
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)