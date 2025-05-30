# Prompt Formatters

def format_mcq_example(example, context_chunks, include_relevant):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = example["choices"]
    lettered_choices = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]

    context_text = "\n".join([f"Document {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = (
        "\nContext:\n" + context_text + "\n"
        f"Question: {example['question']}\n"
        f"Choices:\n" + "\n".join(lettered_choices) +
        "\nAnswer:"
    )

    # Dynamically determine the Reason
    if include_relevant:
        if example.get("explanation", "").strip():
            reason = example["explanation"]
        else:
            reason = "[Retrieved context can be used to explain the answer]"
    else:
        reason = "No sufficient information available."

    # Target always in Reason + Answer format
    target = (
        f"##Answer: {example['correct_answer']}"
        f" ##Reason: {reason}\n"
    )

    return prompt, target

def format_mcq_test_example(example, context_chunks):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = example["choices"]
    lettered_choices = [f"{letters[i]}. {choice}" for i, choice in enumerate(choices)]

    context_text = "\n".join([f"Document {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = (
        "\nContext:\n" + context_text + "\n"
        f"Question: {example['question']}\n"
        f"Choices:\n" + "\n".join(lettered_choices) +
        "\nAnswer:"
    )

    return prompt

# def format_openqa_prompt()

# def format_rag_prompt