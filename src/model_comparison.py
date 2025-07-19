# src/model_comparison.py
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, f1_score

def compare_models(preds_dict: Dict[str, List[int]], y_true: Optional[List[int]] = None) -> pd.DataFrame:
    print('[model_comparison] Comparing models')
    rows = []
    for model, preds in preds_dict.items():
        if y_true is not None:
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds)
        else:
            acc = None
            f1 = None
        rows.append({'Model': model, 'Accuracy': acc, 'F1': f1})
    print('[model_comparison] Model comparison complete')
    return pd.DataFrame(rows) 