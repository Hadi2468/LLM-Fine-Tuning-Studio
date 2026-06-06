import os
import subprocess

REPO_URL = "https://github.com/YOUR_USERNAME/LLM-Fine-Tuning-Studio.git"
PROJECT_DIR = "/content/project"

if not os.path.exists(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])

os.chdir(PROJECT_DIR)

env = os.environ.copy()
env["PYTHONPATH"] = PROJECT_DIR

subprocess.run(["pip", "install", "-r", "requirements.txt"])

subprocess.run(
    ["python", "scripts/run_training.py"],
    env=env
)