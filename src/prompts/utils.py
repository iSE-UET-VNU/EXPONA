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
    return f"""
You are given a task of {config['description']}.

Please generate a Python labeling function according to the following instructions:
{config['instructions']}
"""

def create_system_prompt() -> str:
    return """
Your task is to generate a Python labeling function based on the provided instructions.
The generated function should be wise, creative, and general.
Please only output the full Python code in a code block. Do not write any explanation.
If your function uses any external libraries you must include the necessary import statements.
"""


def upsert_dataset_config(dataset_name: str, description: str, labels=None, multi: bool = False, write_back: bool = True):
    if multi:
        if labels is None:
            raise ValueError("labels must be provided for multi configs and be a list")
        new_entries = []
        for item in labels:
            if isinstance(item, dict):
                desc = item.get("description")
                instr = item.get("instructions")
            else:
                desc = f"text is about {item} or not"
                instr = (
                    f"If the text is not about {item}, function returns 0.\n"
                    f"If the text is about {item}, function returns 1.\n"
                    f"If the text cannot be categorized, function returns -1.\n\n"
                    "function signature: def label_function(text)"
                )
            new_entries.append({"description": desc, "instructions": instr})

        configs_multi[dataset_name] = new_entries
        _MULTI_DATA["configs"] = configs_multi
        if write_back:
            with _MULTI_CFG_PATH.open("w", encoding="utf-8") as f:
                yaml.safe_dump(_MULTI_DATA, f, allow_unicode=True, sort_keys=False)
        return new_entries
    else:
        if isinstance(labels, str):
            instr = labels
        elif isinstance(labels, (list, tuple)):
            lines = []
            for i, lab in enumerate(labels):
                lines.append(f"If the text is about {lab}, function returns {i}.")
            lines.append("If the text cannot be categorized, function returns -1.")
            lines.append("")
            lines.append("function signature: def label_function(text)")
            instr = "\n".join(lines)
        else:
            raise ValueError("labels must be None, a string, or a list/tuple of label names for single configs")

        cfg = {"description": description, "instructions": instr}
        configs[dataset_name] = cfg
        _SINGLE_DATA["configs"] = configs
        if write_back:
            with _SINGLE_CFG_PATH.open("w", encoding="utf-8") as f:
                yaml.safe_dump(_SINGLE_DATA, f, allow_unicode=True, sort_keys=False)
        return cfg