from dataclasses import dataclass, field
from pathlib import Path
import time

@dataclass
class Config:
    cur_time: float = time.time()
    data_home: Path = Path("data")
    dataset: str = "youtube"
    train_size: float = 0.8
    dev_size: float = 0.5
    version: str = "v9"
    verbose: bool = False
    model: str = "gpt-4.1"
    temperature: float = 1
    top_p: float = 0.8
    max_token: int = 10000
    min_lf_per_type: int = 5
    alpha: float = 0.9
    labeled_percentage: float = 0.3
    beta: float = 0.1
    max_patience: int = 3
    eval_times: int = 3
    run_saved_lfs: bool = True
    eval_metric: str = "f1_weighted"
    labeling_models: list[str] = field(default_factory=lambda: ["Snorkel"])
    end_models: list[str] = field(default_factory=lambda: ["MLP"])
    provider: str = "openai"
    api_key: str = ""