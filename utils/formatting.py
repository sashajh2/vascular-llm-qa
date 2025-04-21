# Prompt Formatters

def format_mcq_example(example):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = example["choices"]
    lettered_choices = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]

    explanation = example.get("explanation", "[explanation]")
    source = example.get("source", "[source]")

    prompt = f"Question: {example['question']}\nChoices:\n" + "\n".join(lettered_choices) + "\nAnswer:"

    target = f" {example['correct_answer']} - {explanation} - {source}"
    return prompt, target


# def format_openqa_prompt()

# def format_rag_prompt