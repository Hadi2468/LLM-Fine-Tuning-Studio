import yaml
import os

def load_config():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    config_path = os.path.join(ROOT_DIR, "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ConfigWrapper:
    def __init__(self, config_dict):
        self.config = config_dict

    @property
    def model_name(self):
        return self.config["model"]["name"]

    @property
    def training(self):
        return self.config["training"]