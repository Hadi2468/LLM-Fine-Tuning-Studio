# Dataset loader for LLM fine-tuning, utilizing Hugging Face's datasets library

   
import json
from datasets import Dataset

from src.dataset.validator import validate_dataset
from src.dataset.formatter import format_example




def load_dataset(dataset_path: str):

    validate_dataset(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []

    for row in data:
        formatted_data.append(
            {
                "text": format_example(row)
            }
        )

    return Dataset.from_list(formatted_data)