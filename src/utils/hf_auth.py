# Hugging Face authentication utility

from huggingface_hub import login
import os


def hf_login():

    token = os.getenv("HF_TOKEN")

    if token:
        login(token=token)