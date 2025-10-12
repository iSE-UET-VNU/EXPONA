import logging
from pathlib import Path
from typing import Optional, Union, List
from .base import ProbLF, LFType
from wrench.dataset.basedataset import BaseDataset

logger = logging.getLogger(__name__)

class StructuralLF(ProbLF):
    def __init__(
        self,
        lf=None,
        eval_metric: str="f1_weighted",
        saved_path: Optional[Union[str, Path]] = None,
        lf_path: Path = Path("label_function"),
    ):
        super().__init__(
            lf_path=lf_path,
            lf_type=LFType.STRUCTURAL,
            lf=lf,
            saved_path=saved_path,
            eval_metric=eval_metric,
        )

    def find_threshold(self, labeled_dataset: BaseDataset, unlabeled_dataset: BaseDataset, beta=0.1) -> None:
        self._find_threshold(labeled_dataset.features_2, unlabeled_dataset.features_2, labeled_dataset.labels, beta)
    
    def get_labels(self, dataset: BaseDataset) -> List:
        features = dataset.features_2
        final_preds = self._get_labels(features)
        return final_preds