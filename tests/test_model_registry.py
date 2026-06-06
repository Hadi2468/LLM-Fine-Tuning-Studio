# Test for the model registry functionality

from src.models.model_registry import (
    get_all_models,
    get_model_path,
    get_model_group,
)


def test_registry():

    print()

    print(get_all_models())

    print()

    print(
        get_model_path(
            "Llama-3.1-8B"
        )
    )

    print()

    print(
        get_model_group(
            "Llama-3.1-8B"
        )
    )


if __name__ == "__main__":

    test_registry()