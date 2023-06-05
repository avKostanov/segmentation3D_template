from pathlib import Path
from typing import Dict

import yaml


def compose_config(config_path: Path) -> Dict:
    config_full = {}
    for config_file in config_path.iterdir():
        with open(config_file, 'r') as f:
            config_full.update(yaml.safe_load(f))

    return config_full
