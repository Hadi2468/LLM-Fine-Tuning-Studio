import os
import subprocess
import sys

REPO_URL = "https://github.com/YOUR_USERNAME/LLM-Fine-Tuning-Studio.git"
PROJECT_DIR = "/content/project"

# clone
if not os.path.exists(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])

os.chdir(PROJECT_DIR)

# install deps
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# force python to see src
env = os.environ.copy()
env["PYTHONPATH"] = PROJECT_DIR

# run training from ROOT (NOT scripts folder)
subprocess.run(
    ["python", "scripts/run_training.py"],
    env=env,
    cwd=PROJECT_DIR
)