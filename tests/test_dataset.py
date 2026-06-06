from src.dataset.loader import load_dataset

# # Test 1
# data = load_dataset(
#     "datasets/sample.json"
# )

# print(data[0])

# # Test 2
# dataset = load_dataset("datasets/sample.json")

# print(dataset)
# print(dataset[0])
# print(dataset.column_names)

# Test 3
from transformers import AutoTokenizer

dataset = load_dataset("datasets/sample.json")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B"
)

sample = dataset[0]["text"]

tokens = tokenizer(
    sample,
    truncation=True,
    max_length=512
)

print(tokens.keys())
print(len(tokens["input_ids"]))


# Test 4
tokenized_dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        max_length=512
    ),
    batched=True
)

print(tokenized_dataset)
print(tokenized_dataset.column_names)
print(tokenized_dataset[0].keys())
