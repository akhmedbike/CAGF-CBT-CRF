from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

@dataclass
class TaskMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)

def compute_classification_metrics(y_true: List[int], y_pred: List[int], average: str='macro') -> TaskMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(f'length mismatch: {len(y_true)} true vs {len(y_pred)} pred')
    if len(y_true) == 0:
        raise ValueError('cannot compute metrics on an empty sequence')
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return TaskMetrics(accuracy=float(acc), precision=float(precision), recall=float(recall), f1=float(f1))

def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> TaskMetrics:
    if y_true.shape != y_pred.shape:
        raise ValueError(f'shape mismatch: {y_true.shape} vs {y_pred.shape}')
    exact_match = float((y_true == y_pred).all(axis=1).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return TaskMetrics(accuracy=exact_match, precision=float(precision), recall=float(recall), f1=float(f1))