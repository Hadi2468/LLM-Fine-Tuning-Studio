import os
import subprocess

REPO_URL = "https://github.com/Hadi2468/LLM-Fine-Tuning-Studio.git"
PROJECT_DIR = "/content/project"

if not os.path.exists(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])

os.chdir(PROJECT_DIR)

subprocess.run(["git", "pull"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# IMPORTANT: run from ROOT context
subprocess.run(
    ["python", "scripts/run_training.py"],
    cwd=PROJECT_DIR
)