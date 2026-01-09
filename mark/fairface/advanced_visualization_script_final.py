# Author: Mark Smithson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

SAVE_DIR = "advanced_stats"
os.makedirs(SAVE_DIR, exist_ok=True)

# For Yunet analysis just change this
backend = 'OpenCV'

# Analysis Functions
def generate_fragility_heatmap(df, attribute, metric='F1'):
    subset = df[df['Attribute'] == attribute].copy()
    
    # Separate baseline and augmentations
    baseline = subset[subset['Dataset'] == 'Baseline'].set_index('Class')[metric]
    augs = [c for c in subset['Dataset'].unique() if c.startswith('Aug_')]
    
    heatmap_data = []
    
    for aug in augs:
        aug_name = aug.replace('Aug_', '')
        aug_vals = subset[subset['Dataset'] == aug].set_index('Class')[metric]
        
        change = (aug_vals - baseline)

        change.name = aug_name
        heatmap_data.append(change)
        
    if not heatmap_data:
        return None
        
    heatmap_df = pd.concat(heatmap_data, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, center=0, cmap="RdBu", fmt=".2f", cbar_kws={'label': f'Change in {metric} (Absolute)'})
    plt.title(f'{attribute} Fragility Heatmap: Change in {metric} vs Baseline')
    plt.tight_layout()

    return plt

def generate_stability_boxplot(df, attribute, metric='F1'):
    subset = df[(df['Attribute'] == attribute) & (df['Dataset'].str.startswith('Aug_'))]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=subset, x='Class', y=metric, palette="Set3")
    sns.stripplot(data=subset, x='Class', y=metric, color=".3", linewidth=1, size=4)
    
    # Add baseline line to compare with augmentations
    baseline = df[(df['Attribute'] == attribute) & (df['Dataset'] == 'Baseline')]
    
    # Plot baseline as red dashed line
    classes = sorted(subset['Class'].unique())
    for i, cls in enumerate(classes):
        base_val = baseline[baseline['Class'] == cls][metric].values
        if len(base_val) > 0:
            plt.hlines(y=base_val[0], xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='--', linewidth=2, label='Baseline' if i == 0 else "")
            
    plt.legend()
    plt.title(f'{attribute} Stability Analysis: Variation across Augmentations')
    plt.tight_layout()
    return plt

def calculate_cv_stats(df, attribute, metric='F1'):
    # Calculates Coefficient of Variation for each class across augmentations.
    subset = df[(df['Attribute'] == attribute) & (df['Dataset'].str.startswith('Aug_'))]
    
    stats = subset.groupby('Class')[metric].agg(['mean', 'std'])
    stats['CV'] = stats['std'] / stats['mean']
    return stats.sort_values('CV', ascending=False) # Sort by robustness


df = pd.read_csv('granular_results/granular_metrics_all.csv')

df_backend = df[df['Backend'] == backend]

# Race Analysis
hm_race = generate_fragility_heatmap(df_backend, 'Race')
if hm_race:
    hm_race.savefig(f"{SAVE_DIR}/Race_Fragility_Heatmap.png")
    plt.close()

bp_race = generate_stability_boxplot(df_backend, 'Race')
bp_race.savefig(f"{SAVE_DIR}/Race_Stability_Boxplot.png")
plt.close()

# Gender Analysis
hm_gender = generate_fragility_heatmap(df_backend, 'Gender')
if hm_gender:
    hm_gender.savefig(f"{SAVE_DIR}/Gender_Fragility_Heatmap.png")
    plt.close()
    
# Age Analysis (Focus on 40-49 Anomaly)
hm_age = generate_fragility_heatmap(df_backend, 'Age')
if hm_age:
    hm_age.savefig(f"{SAVE_DIR}/Age_Fragility_Heatmap.png")
    plt.close()
    
# Statistical Summary Table
race_stats = calculate_cv_stats(df_backend, 'Race')
age_stats = calculate_cv_stats(df_backend, 'Age')

stats_path = f"{SAVE_DIR}/stability_metrics.csv"
with open(stats_path, 'w') as f:
    f.write("--- Race Stability (CV) ---\n")
    race_stats.to_csv(f)
    f.write("\n--- Age Stability (CV) ---\n")
    age_stats.to_csv(f)

print(race_stats)
