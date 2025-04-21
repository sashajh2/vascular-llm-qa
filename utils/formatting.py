# Prompt Formatters

def format_mcq_example(example):
    # Convert structured example to a prompt → target format
    prompt = f"""Question: {example['question']}
Choices:
A. {example['choices'][0]}
B. {example['choices'][1]}
C. {example['choices'][2]}
D. {example['choices'][3]}
Answer:"""
    target = f" {example['correct_answer']} - {example['explanation']} - {example['source']}"
    return prompt, target

# def format_openqa_prompt()

# def format_rag_prompt