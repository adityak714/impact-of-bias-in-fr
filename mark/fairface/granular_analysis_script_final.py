# Author: Mark Smithson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from analysis_helpers_final import load_and_parse_data
from metric_helpers_final import get_per_class_metrics, get_accuracy_by_subgroup

RESULTS_DIR = "granular_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

files = {
    'OpenCV_Baseline': 'opencv_deepface_gender_predictions.csv',  # This includes gender and race predictions
    'Yunet_Baseline': 'yunet_deepface_gender_predictions.csv',  # This includes gender and race predictions
    'OpenCV_Age_Baseline': 'opencv_deepface_age_predictions.csv',
    'Yunet_Age_Baseline': 'yunet_deepface_age_predictions.csv',
    'OpenCV_Augmented': 'opencv_deepface_predictions.csv',
    'Yunet_Augmented': 'yunet_deepface_predictions.csv'
}

data = {}

# Load Baselines (Gender/Race)
base_gr_opencv = load_and_parse_data(files['OpenCV_Baseline'], is_baseline_dict=True)
base_gr_yunet = load_and_parse_data(files['Yunet_Baseline'], is_baseline_dict=True)

# Load Baselines (Age)
age_opencv = load_and_parse_data(files['OpenCV_Age_Baseline'], is_baseline_dict=False)
age_yunet = load_and_parse_data(files['Yunet_Age_Baseline'], is_baseline_dict=False)

# We merge Age pred into Gender/Race base
data['OpenCV_Base'] = pd.merge(base_gr_opencv, age_opencv[['filename', 'age_range_pred']], on='filename', how='inner')
data['Yunet_Base'] = pd.merge(base_gr_yunet, age_yunet[['filename', 'age_range_pred']], on='filename', how='inner')

data['OpenCV_Aug'] = load_and_parse_data(files['OpenCV_Augmented'])
data['Yunet_Aug'] = load_and_parse_data(files['Yunet_Augmented'])

def run_full_benchmark(backend_name, df_base, df_aug):
    
    attributes = [
        ('Gender', 'gender_true', 'gender_pred'),
        ('Race', 'race_true', 'race_pred'),
        ('Age', 'age_range_true', 'age_range_pred')
    ]
    
    all_metrics = []
    
    for attr, true_col, pred_col in attributes:
        base_metrics = get_per_class_metrics(df_base, true_col, pred_col)
        base_metrics['Dataset'] = 'Baseline'
        base_metrics['Backend'] = backend_name
        base_metrics['Attribute'] = attr
        
        # Augmented Performance (Overall)
        aug_metrics = get_per_class_metrics(df_aug, true_col, pred_col)
        aug_metrics['Dataset'] = 'Augmented_All'
        aug_metrics['Backend'] = backend_name
        aug_metrics['Attribute'] = attr
        
        all_metrics.append(pd.concat([base_metrics, aug_metrics]))
        
        # Per-Augmentation Analysis
        for aug_type in df_aug['augmentation'].unique():
            if pd.isna(aug_type) or aug_type == 'original': continue
            
            sub_aug = df_aug[df_aug['augmentation'] == aug_type]
            sub_metrics = get_per_class_metrics(sub_aug, true_col, pred_col)
            sub_metrics['Dataset'] = f'Aug_{aug_type}'
            sub_metrics['Backend'] = backend_name
            sub_metrics['Attribute'] = attr
            
            all_metrics.append(sub_metrics)

    return pd.concat(all_metrics)

# Run Benchmarks
res_opencv = run_full_benchmark('OpenCV', data['OpenCV_Base'], data['OpenCV_Aug'])
res_yunet = run_full_benchmark('Yunet', data['Yunet_Base'], data['Yunet_Aug'])

final_results = pd.concat([res_opencv, res_yunet])

csv_path = f"{RESULTS_DIR}/granular_metrics_all.csv"
final_results.to_csv(csv_path, index=False)

def plot_comparisons(df):
    # Filter for Baseline vs Augmented_All
    # Plotting Precision/Recall for each class
    subset = df[df['Dataset'].isin(['Baseline', 'Augmented_All'])]
    
    for backend in df['Backend'].unique():
        for attr in df['Attribute'].unique():
            data_slice = subset[(subset['Backend'] == backend) & (subset['Attribute'] == attr)]
            
            if len(data_slice) == 0: continue
            
            # Melt for sns
            melted = data_slice.melt(id_vars=['Class', 'Dataset'], value_vars=['Precision', 'Recall'], 
                                   var_name='Metric', value_name='Score')
            
            plt.figure(figsize=(10, 6))
            
            # Using catplot for faceted bar chart
            g = sns.catplot(data=melted, x='Class', y='Score', hue='Dataset', col='Metric', kind='bar', height=5, aspect=1.2)
            g.fig.suptitle(f'{backend} - {attr} Performance: Baseline vs Augmented', y=1.02)
            plt.xticks(rotation=45)
            
            out_file = f"{RESULTS_DIR}/{backend}_{attr}_comparison.png"
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()

plot_comparisons(final_results)

pivoted = final_results.pivot_table(index=['Backend', 'Attribute', 'Class'], 
                                    columns='Dataset', 
                                    values='F1')

aug_cols = [c for c in pivoted.columns if 'Aug_' in c and c != 'Augmented_All']
best_augs = []

for idx, row in pivoted.iterrows():
    base_score = row['Baseline']
    for col in aug_cols:
        score = row[col]
        if score > base_score:
             best_augs.append({
                 'Backend': idx[0],
                 'Attribute': idx[1],
                 'Class': idx[2],
                 'Augmentation': col,
                 'Baseline_F1': base_score,
                 'Aug_F1': score,
                 'Impovement': score - base_score
             })


# Save every augmentation win over baseline
if len(best_augs) > 0:
    wins_df = pd.DataFrame(best_augs).sort_values('Impovement', ascending=False)
    wins_path = f"{RESULTS_DIR}/augmentation_wins.csv"
    wins_df.to_csv(wins_path, index=False)