# Converts raw dataset examples into the format required for fine-tuning

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}
"""

def format_example(example):

    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        response=example["response"]
    )