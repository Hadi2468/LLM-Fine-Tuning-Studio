def load_config():
    import yaml
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)