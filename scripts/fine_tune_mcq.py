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

def load_dataset_from_jsonl(jsonl_path):
    """Load and parse a JSONL dataset for instruction fine-tuning."""
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def process_example(example, tokenizer, P=0.8):
    # Randomly decide to include golden doc or only distractors
    include_golden = random.random() < P
    context_chunks = retrieve_docs(example["question"], include_golden)
    
    # Format prompt/target using retrieved docs
    prompt, target = format_mcq_example(example, context_chunks, include_golden)
    
    # Tokenize
    tokenized = tokenizer(
        prompt + target,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized



def fine_tune(model_name_or_path, data_path, output_dir, num_train_epochs=3):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Load and tokenize dataset
    dataset = load_dataset_from_jsonl(data_path)
    tokenized_dataset = dataset.map(lambda x: process_example(x, tokenizer))

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        eval_strategy="no",
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

    # Train and save
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
