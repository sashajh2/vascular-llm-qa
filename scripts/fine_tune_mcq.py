import argparse
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from utils.formatting import format_mcq_example
import random
from retriever.retrieval import retrieve_docs
from utils.data_utils import train_test_split

def load_dataset_from_jsonl(jsonl_path):
    """Load and parse a JSONL dataset for instruction fine-tuning."""
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def generate_dataset(dataset, tokenizer):
    """Generate golden and distractor examples for each question."""
    examples = []

    for example in dataset:
        # Golden + distractors version
        context_chunks_golden = retrieve_docs(example["question"], include_golden=True)
        prompt_golden, target_golden = format_mcq_example(example, context_chunks_golden, include_golden=True)
        tokenized_golden = tokenizer(
            prompt_golden + target_golden,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized_golden["labels"] = tokenized_golden["input_ids"].copy()
        examples.append(tokenized_golden)

        # Distractors only version
        context_chunks_distractor = retrieve_docs(example["question"], include_golden=False)
        prompt_distractor, target_distractor = format_mcq_example(example, context_chunks_distractor, include_golden=False)
        tokenized_distractor = tokenizer(
            prompt_distractor + target_distractor,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized_distractor["labels"] = tokenized_distractor["input_ids"].copy()
        examples.append(tokenized_distractor)

    return examples



def fine_tune(model_name_or_path, data_path, output_dir, num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    dataset = load_dataset_from_jsonl(data_path)
    train_test_split(dataset)

    # NEW: Generate full training set
    tokenized_examples = generate_dataset(dataset, tokenizer)
    tokenized_dataset = Dataset.from_list(tokenized_examples)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")



if __name__ == "__main__":
    # Limit training epochs to avoid overfitting
    # Use dropout and regularization
    # Source generation and explaations will likely be shaky at first!
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    fine_tune(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
    )
