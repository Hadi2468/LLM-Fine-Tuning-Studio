from pathlib import Path
import json


def validate_dataset(dataset_path: str):

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Dataset must be a list of examples."
        )

    required_fields = {
        "instruction",
        "response"
    }

    for idx, row in enumerate(data):

        missing = required_fields - row.keys()

        if missing:
            raise ValueError(
                f"Row {idx} missing fields: {missing}"
            )

    return True