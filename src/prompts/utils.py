import re
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent
_SINGLE_CFG_PATH = _BASE / "config.yml"
_MULTI_CFG_PATH = _BASE / "config_multi.yml"

def _load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_SINGLE_DATA = _load_yaml(_SINGLE_CFG_PATH)
_MULTI_DATA = _load_yaml(_MULTI_CFG_PATH)

configs = _SINGLE_DATA.get("configs", {})
configs_multi = _MULTI_DATA.get("configs", {})

def extract_response(response):
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) == 0:
        return response
    return matches[0]

def create_user_prompt(dataset_name: str, label_number: int = None) -> str:
    if label_number is not None:
        entries = configs_multi.get(dataset_name)
        if entries is None:
            raise ValueError(f"Dataset '{dataset_name}' is not supported in configs_multi.")
        try:
            config = entries[int(label_number)]
        except (IndexError, ValueError):
            raise ValueError(f"label_number {label_number} out of range for dataset '{dataset_name}'.")
    else:
        config = configs.get(dataset_name)
    if not config:
        raise ValueError(f"Dataset '{dataset_name}' is not supported in configs.")
    return (
    f"You are given a {config['description']} task.\n"
    f"Write a Python labeling function that follows these instructions:\n"
    f"{config['instructions']}\n"
    "Please only output the full Python code in a code block. Do not write any explanation.\n"
    "If your function uses any external libraries you must include the necessary import statements.\n"
)

def create_system_prompt(dataset_name: str, label_number: int = None) -> str:
    if label_number is not None:
        entries = configs_multi.get(dataset_name)
        if entries is None:
            raise ValueError(f"Dataset '{dataset_name}' is not supported in configs_multi.")
        try:
            config = entries[int(label_number)]
        except (IndexError, ValueError):
            raise ValueError(f"label_number {label_number} out of range for dataset '{dataset_name}'.")
    else:
        config = configs.get(dataset_name)

    if not config:
        raise ValueError(f"Dataset '{dataset_name}' is not supported in configs.")
    return (
        f"You are a helpful assistant for a {config['description']} task.\n"
        "Your task is to generate a Python labeling function based on the provided instructions.\n"
        "The generated function should be wise, creative, and general."
        "Always return only the function code in a code block."
    )