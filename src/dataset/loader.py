# Dataset loader for LLM fine-tuning, utilizing Hugging Face's datasets library


from datasets import Dataset
import pandas as pd


class DatasetLoader:

    @staticmethod
    def from_dataframe(df):

        return Dataset.from_pandas(df)