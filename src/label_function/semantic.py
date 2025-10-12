import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from .base import BaseLF
from wrench.dataset.basedataset import BaseDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

def get_unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    directory = base_path.parent

    index = 1
    while True:
        new_name = f"{stem}_{index}{suffix}"
        new_path = directory / new_name
        if not new_path.exists():
            return new_path
        index += 1

class SemanticLF(BaseLF):
    def __init__(
        self,
        lf,
        features_combs,
        eval_metric: str,
        saved_path: Optional[Union[str, Path]] = None,
        *args,
        **kwargs
    ):
        if lf is None and saved_path is None:
            raise ValueError("Must provide either a lf model or a saved_path to load from.")
        
        self.lf = lf
        if saved_path:
            self.load_lf(saved_path)
        self.features_combs = features_combs
        self.threshold = None

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

    def save(self, save_dir: Union[str, Path], filename: str = "model.pkl") -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_path = save_dir / filename
        model_path = get_unique_path(base_path)
        
        joblib.dump(self.lf, model_path)
        print(f"Heuristic model saved to {model_path}")
        return model_path
    
    def load_lf(self, path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        print(f"Loading lf model from {path}")
        return joblib.load(path)

    def find_threshold(self, labeled_dataset: BaseDataset, unlabeled_dataset: BaseDataset, beta=0.1) -> None:
        eps = 1e-8
    
        X_lab = labeled_dataset.features[:, self.features_combs]
        X_unlab = unlabeled_dataset.features[:, self.features_combs]
        y_lab = np.asarray(labeled_dataset.labels)
    
        probs_lab = self.lf.predict_proba(X_lab)
        preds_lab = probs_lab.argmax(axis=1)
        conf_lab = probs_lab.max(axis=1)
    
        probs_unlab = self.lf.predict_proba(X_unlab) if len(X_unlab) > 0 else None
        conf_unlab = probs_unlab.max(axis=1) if probs_unlab is not None else np.array([])
    
        thresholds = np.linspace(0.0, 1.0, 101)
        best_f_beta = -1.0
        best_t = 0.0
    
        for t in thresholds:
            mask_lab = conf_lab >= t
            if mask_lab.sum() == 0:
                continue
            precision_proxy = self.score_funct(preds_lab[mask_lab], y_lab[mask_lab])
    
            if conf_lab.size > 0:
                recall_proxy = (conf_unlab >= t).mean()
            else:
                recall_proxy = 0.0
    
            f_beta = (1 + beta**2) * (precision_proxy * recall_proxy) / (beta**2 * precision_proxy + recall_proxy + eps)
    
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

    def get_labels(self, dataset: BaseDataset) -> List:
        probs = self.lf.predict_proba(dataset.features[:, self.features_combs])
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        final_preds = np.full_like(preds, fill_value=-1)
        mask = confidences >= self.threshold
        final_preds[mask] = preds[mask]
        return final_preds