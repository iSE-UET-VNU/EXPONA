import logging
from pathlib import Path
from typing import List, Optional, Union
from abc import ABC
from enum import Enum
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class LFType(Enum):
    SURFACE = "surface"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"

class BaseLF(ABC):
    def __init__(self, lf_path: Path, lf_type: LFType = None):
        lf_path.mkdir(parents=True, exist_ok=True)
        self.lf_path = lf_path / lf_type.value
        self.lf_path.mkdir(parents=True, exist_ok=True)
        self.index = -1
        self.lf_type = lf_type

    def get_index(self) -> int:
        if self.index == -1:
            raise ValueError("The lf is not saved.")

        return self.index

    def find_available_index(self):
        idx = 0
        saved_path = self.lf_path / f"lf_{idx}.py"
        while saved_path.exists():
            idx += 1
            saved_path = self.lf_path / f"lf_{idx}.py"
        
        return idx

    def get_eval_metric(self, eval_metric):
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

class ProbLF(BaseLF):
    def __init__(
        self,
        lf=None,
        eval_metric: str="f1_weighted",
        saved_path: Optional[Union[str, Path]] = None,
        lf_type: LFType = None,
        lf_path: Path = Path("label_function"),
        *args,
        **kwargs
    ):
        super().__init__(lf_path, lf_type)
        if lf is None and saved_path is None:
            raise ValueError("Must provide either a lf model or a saved_path to load from.")
        
        if saved_path:
            self.lf = self.load_lf(saved_path)
        else:
            self.lf = lf
        self.threshold = None

        self.get_eval_metric(eval_metric)
    
    def find_available_index(self):
        idx = 0
        saved_path = self.lf_path / f"lf_{idx}.pkl"
        while saved_path.exists():
            idx += 1
            saved_path = self.lf_path / f"lf_{idx}.pkl"
        
        return idx

    def save(self) -> Path:
        model_path = self.lf_path / f"lf_{self.find_available_index()}.pkl"
        joblib.dump(self.lf, model_path)
        logger.info(f"LF saved to {model_path}")
        return model_path

    def load_lf(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"LF does not exist: {path}")
        logger.info(f"Loading lf from {path}")
        return joblib.load(path)

    def _find_threshold(self, labeled_features, unlabeled_features, labels, beta=0.1):
        X_lab = labeled_features
        y_lab = np.asarray(labels)
        probs_lab = self.lf.predict_proba(X_lab)
        preds_lab = probs_lab.argmax(axis=1)
        conf_lab = probs_lab.max(axis=1)
        
        X_unlab = unlabeled_features
        probs_unlab = self.lf.predict_proba(X_unlab) if len(X_unlab) > 0 else None
        conf_unlab = probs_unlab.max(axis=1)
        
        thresholds = np.linspace(0.0, 1.0, 101)
        best_f_beta = -1.0
        best_t = 0.0
    
        for t in thresholds:
            mask_lab = conf_lab >= t
            if mask_lab.sum() == 0:
                continue
            precision_proxy = self.score_funct(preds_lab[mask_lab], y_lab[mask_lab])
    
            if conf_unlab.size > 0:
                recall_proxy = (conf_unlab >= t).mean()
            else:
                recall_proxy = 0.0
    
            f_beta = (1 + beta**2) * (precision_proxy * recall_proxy) / (beta**2 * precision_proxy + recall_proxy + 1e-8)
    
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_t = t
    
        mask_lab_final = conf_lab >= best_t
        if mask_lab_final.sum() > 0:
            perf = self.score_funct(preds_lab[mask_lab_final], y_lab[mask_lab_final])
        else:
            perf = 0.0
    
        cov_unlab_final = float((conf_unlab >= best_t).mean()) if conf_unlab.size > 0 else 0.0
    
        self.threshold = float(best_t)
        self.estimated_fbeta = float(best_f_beta)
        self.estimated_performance = float(perf)   
        self.estimated_coverage = float(cov_unlab_final) 
    
    def _get_labels(self, features) -> List:
        probs = self.lf.predict_proba(features)
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        final_preds = np.full_like(preds, fill_value=-1)
        mask = confidences >= self.threshold
        final_preds[mask] = preds[mask]
        return final_preds