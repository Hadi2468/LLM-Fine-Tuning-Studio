import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.utils.config import load_config
from src.core.orchestrator import TrainingOrchestrator

def main():
    print("🚀 Starting LLM Fine-Tuning Studio")

    config = load_config()

    orchestrator = TrainingOrchestrator(config)

    orchestrator.run()

    print("✅ Training finished successfully")


if __name__ == "__main__":
    main()