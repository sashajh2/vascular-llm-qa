# Prompt Formatters

def format_mcq_example(example, context_chunks):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = example["choices"]
    lettered_choices = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]
    
    # Insert retrieved context here
    context_text = "\n".join([f"Document {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])

    # Prompt includes context
    prompt = (
        f"Question: {example['question']}\n"
        f"Choices:\n" + "\n".join(lettered_choices) +
        "\nContext:\n" + context_text + 
        "\nAnswer:"
    )

    # Target is still: Answer - Explanation
    # You may remove explanation/source placeholders if you want purely context-driven training
    target = f" {example['correct_answer']} - [explanation]"

    return prompt, target



# def format_openqa_prompt()

# def format_rag_prompt