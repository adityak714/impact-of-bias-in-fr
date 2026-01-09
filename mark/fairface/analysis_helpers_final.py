# Author: Mark Smithson

import pandas as pd
import numpy as np

AGE_MAP = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 
           5: "40-49", 6: "50-59", 7: "60-69", 8: "+70"}

RACE_MAP = {
    0: "asian",           
    1: "indian",          
    2: "black",           
    3: "white",           
    4: "middle eastern",  
    5: "latino hispanic", 
    6: "asian"            
}

GENDER_MAP = {0: "Male", 1: "Female"}

def parse_deepface_dict(val, is_gender=True):
    try:
        s = str(val)
        if is_gender:
            man_pos = s.find("'Man'")
            woman_pos = s.find("'Woman'")
            
            # Simple fallback: extract number after 'Man'
            man_part = s.split("'Man':")[1].split('(')[1].split(')')[0]
            woman_part = s.split("'Woman':")[1].split('(')[1].split(')')[0]
            
            man_score = float(man_part)
            woman_score = float(woman_part)
            
            return 'Male' if man_score > woman_score else 'Female'
        else:
            s = s.strip("{}")
            pairs = s.split(", '")
            
            max_score = -1.0
            max_label = None
            
            for pair in pairs:
                if ":" not in pair: continue
                
                parts = pair.split(":")
                label = parts[0].strip("' ")
                score_str = parts[1]
                
                if "float32" in score_str:
                    score = float(score_str.split("(")[1].split(")")[0])
                else:
                    score = float(score_str)
                    
                if score > max_score:
                    max_score = score
                    max_label = label
            
            return max_label
    except Exception as e:
        return None

def extract_labels_from_filename(filename):
    clean_name = filename.replace('.jpg', '')
    parts = clean_name.split('_')

    race_code = int(parts[2])
    age_code = int(parts[3])
    gender_code = int(parts[4])
    
    if len(parts) > 6:
        augmentation = "_".join(parts[6:])
    else:
        augmentation = "original"
        
    return {
        'race_code': race_code,
        'age_code': age_code,
        'gender_code': gender_code,
        'race_true': RACE_MAP.get(race_code, 'unknown'),
        'age_range_true': AGE_MAP.get(age_code, 'unknown'),
        'gender_true': GENDER_MAP.get(gender_code, 'unknown'),
        'augmentation': augmentation
    }

def load_and_parse_data(filepath, is_baseline_dict=False):
    df = pd.read_csv(filepath)
    
    extracted = df['filename'].apply(extract_labels_from_filename)
    df_meta = pd.DataFrame(extracted.tolist())
    df = pd.concat([df, df_meta], axis=1)
    
    # Parse Baseline Dict
    if is_baseline_dict:
        if 'gender' in df.columns:
            df['gender_pred'] = df['gender'].apply(lambda x: parse_deepface_dict(x, is_gender=True))
        if 'race' in df.columns:
            df['race_pred'] = df['race'].apply(lambda x: parse_deepface_dict(x, is_gender=False))
    
    if 'predicted_gender' in df.columns:
        df['gender_pred'] = df['predicted_gender'].apply(lambda x: 'Male' if x == 'Man' else 'Female')
    
    if 'predicted_race' in df.columns:
        df['race_pred'] = df['predicted_race']
        
    if 'predicted_age' in df.columns:
        # Map predicted age float to range
        # Define helper for age range mapping
        def get_age_range(age_float):
            try:
                age = float(age_float)
                if age <= 2: return "0-2"
                if age <= 9: return "3-9"
                if age <= 19: return "10-19"
                if age <= 29: return "20-29"
                if age <= 39: return "30-39"
                if age <= 49: return "40-49"
                if age <= 59: return "50-59"
                if age <= 69: return "60-69"
                return "+70"
            except: return None
            
        df['age_range_pred'] = df['predicted_age'].apply(get_age_range)
    elif 'age' in df.columns and not is_baseline_dict:
        # Handle integer age column (common in baseline age files)
        def get_age_range_int(age_val):
            try:
                age = float(age_val)
                if age <= 2: return "0-2"
                if age <= 9: return "3-9"
                if age <= 19: return "10-19"
                if age <= 29: return "20-29"
                if age <= 39: return "30-39"
                if age <= 49: return "40-49"
                if age <= 59: return "50-59"
                if age <= 69: return "60-69"
                return "+70"
            except: 
                return None
        
        df['age_range_pred'] = df['age'].apply(get_age_range_int)

    return df