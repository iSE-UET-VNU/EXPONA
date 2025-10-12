import logging
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union
from .base import BaseLF
from wrench.dataset.basedataset import BaseDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class SurfaceLF(BaseLF):
    def __init__(self, raw_code: str = None, eval_metric: str = "f1_weighted",  saved_path: Optional[Union[str, Path]] = None, *args, **kwargs):
    
        if raw_code:
            self.lf = self._raw_code_to_funct(raw_code)
        
        if saved_path:
            self.lf = self.load_lf(saved_path)
            self.saved_path = saved_path
            
        if eval_metric == "f1_weighted":
            self.score_funct = lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")
        elif eval_metric == "f1_macro":
            self.score_funct = lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
        elif eval_metric == "f1_micro":
            self.score_funct = lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")
        elif eval_metric == "roc_auc":
            self.score_funct = lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
        elif eval_metric == "acc":
            self.score_funct = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")

    def save(self, lf_type, info, user_prompt, system_prompt) -> Path:
        saved_path = self._find_saved_path()
        try:
            with open(saved_path, "w", encoding="utf-8") as f:
                f.write("'''")
                f.write(f"User Prompt: {user_prompt}\n")
                f.write(f"System Prompt: {system_prompt}\n")
                f.write(f"Heuristic type: {lf_type}\n")
                f.write(f"Input token: {info.get("input_tokens")}\n")
                f.write(f"Output token: {info.get("output_tokens")}\n")
                f.write(f"Cost: {info.get("cost_usd")}\n")
                f.write("'''\n\n\n")
                f.write(self.raw_code)
            print(f"Label function saved to {saved_path}")
        except Exception as e:
            logging.error(f"Error writing to file: {e}")
        self.saved_path = saved_path
        return saved_path
    
    def load_lf(self, path: Union[str, Path]):
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as file:
            code_str = file.read()
            return self._raw_code_to_funct(code_str)
    
    def get_labels(self, dataset: BaseDataset):
        labels = []
        for data_point in dataset.examples:
            labels.append(self.lf(data_point['text']))
            
        return labels      
    
    def estimate_performance(self, labeled_dataset: BaseDataset, unlabeled_dataset: BaseDataset, beta=0.1):
        eps = 1e-8
        
        pred_labels_lab = np.asarray(self.get_labels(labeled_dataset))
        true_labels_lab = np.asarray(labeled_dataset.labels)
    
        mask_lab = (pred_labels_lab != -1)
        if mask_lab.sum() > 0:
            performance = float(self.score_funct(pred_labels_lab[mask_lab], true_labels_lab[mask_lab]))
        else:
            performance = 0.0
    
        if len(unlabeled_dataset) > 0:
            pred_labels_unlab = np.asarray(self.get_labels(unlabeled_dataset))
            coverage = float((pred_labels_unlab != -1).mean())
        else:
            coverage = 0.0

        f_beta = float((1 + beta**2) * (performance * coverage) / (beta**2 * performance + coverage + eps))
    
        self.estimated_performance = performance
        self.estimated_coverage = coverage
        self.estimated_fbeta = f_beta

    def _raw_code_to_funct(self, code_str: str) -> Optional[Callable]:
        exec_env = {}
        try:
            exec(code_str, exec_env, exec_env)
            lf_func = exec_env.get("label_function")
            if not callable(lf_func):
                logging.error("Function 'label_function' not found or not callable.")
                return None
            return lf_func
        except Exception as e:
            logging.error(f"Error extracting function: {e}")
            return None

    @staticmethod
    def is_runnable_lf(code_str: str, test_input=None) -> bool:
        try:
            compile(code_str, '<string>', 'exec')
        except Exception as e:
            logging.error(f"LF Syntax Error: {e}")
            return False

        try:
            exec_env = {}
            exec(code_str, exec_env, exec_env)

            lf_func = exec_env.get("label_function")
            if not callable(lf_func):
                logging.error("No callable function named 'label_function' found.")
                return False
            if test_input is not None:
                lf_func(test_input)
                
        except Exception as e:
            logging.error(f"LF Runtime Error: {e}")
            return False

        return True
