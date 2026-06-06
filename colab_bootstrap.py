import os
import subprocess

REPO_URL = "https://github.com/Hadi2468/LLM-Fine-Tuning-Studio.git"
PROJECT_DIR = "/content/project"

# 1. Clone repo
if not os.path.exists(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])

os.chdir(PROJECT_DIR)

# 2. Pull latest changes
subprocess.run(["git", "pull"])

# 3. Install dependencies
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# 4. Run training
subprocess.run(["python", "scripts/run_training.py"])