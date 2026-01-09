# Author: Mark Smithson

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics_for_group(df, true_col, pred_col):
    if len(df) == 0:
        return {'Count': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}

    y_true = df[true_col]
    y_pred = df[pred_col]
    
    acc = accuracy_score(y_true, y_pred)
    
    return {'Count': len(df), 'Accuracy': acc}


def get_per_class_metrics(df, true_col, pred_col):
    df = df.dropna(subset=[true_col, pred_col])
    classes = sorted(df[true_col].unique())
    
    metrics = []
    
    for cls in classes:
        true_subset = df[df[true_col] == cls]
        recall = (true_subset[pred_col] == cls).mean() if len(true_subset) > 0 else 0
        
        pred_subset = df[df[pred_col] == cls]
        precision = (pred_subset[true_col] == cls).mean() if len(pred_subset) > 0 else 0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        metrics.append({
            'Class': cls,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support (True)': len(true_subset),
            'Support (Pred)': len(pred_subset)
        })
        
    return pd.DataFrame(metrics)

def get_accuracy_by_subgroup(df, subgroup_col, true_col, pred_col):
    results = []
    groups = sorted(df[subgroup_col].dropna().unique())
    for g in groups:
        sub = df[df[subgroup_col] == g]
        acc = (sub[true_col] == sub[pred_col]).mean()
        results.append({
            'Subgroup': g,
            'Accuracy': acc,
            'Count': len(sub)
        })
    return pd.DataFrame(results)
