import argparse
import json
from transformers import (
    Adafactor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from utils.formatting import format_mcq_example
from utils.data_utils import load_dataset_from_jsonl
import random
from retriever.retrieval import retrieve_docs
from torch.optim.lr_scheduler import LambdaLR


def generate_mcq_train_dataset(dataset, tokenizer):
    """Generate golden and distractor examples for each question."""
    examples = []

    for example in dataset:
        # Golden + distractors version
        context_chunks_relevant = retrieve_docs(example["question"], include_relevant=True)
        prompt_relevant, target_relevant = format_mcq_example(example, context_chunks_relevant, include_relevant=True)
        tokenized_golden = tokenizer(
            prompt_relevant + target_relevant,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized_golden["labels"] = tokenized_golden["input_ids"].copy()
        examples.append(tokenized_golden)

        # Distractors only version
        context_chunks_distractor = retrieve_docs(example["question"], include_relevant=False)
        prompt_distractor, target_distractor = format_mcq_example(example, context_chunks_distractor, include_relevant=False)
        tokenized_distractor = tokenizer(
            prompt_distractor + target_distractor,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized_distractor["labels"] = tokenized_distractor["input_ids"].copy()
        examples.append(tokenized_distractor)

    return examples


def fine_tune(model_name_or_path, train_data_path, output_dir, num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_dataset = load_dataset_from_jsonl(train_data_path)
    tokenized_train_examples = generate_mcq_train_dataset(train_dataset, tokenizer)
    tokenized_train_dataset = Dataset.from_list(tokenized_train_examples)

    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None, 
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=num_train_epochs,
        save_strategy="no",
        logging_steps=200,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")


if __name__ == "__main__":
    # Limit training epochs to avoid overfitting
    # Use dropout and regularization
    # Source generation and explanations will likely be shaky at first!
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    fine_tune(
        model_name_or_path=args.model_name_or_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
    )
