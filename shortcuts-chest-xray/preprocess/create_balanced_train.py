import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

base_filename = 'balanced_dataset' 

def extract_patient_id(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('patient'):
            return part
    return None

def get_patient_ids_to_remove(df_ref, df_subset):
    subset_patient_ids = df_subset['Path'].apply(extract_patient_id).unique()
    ids_to_remove = []
    for patient_id in subset_patient_ids:
        # Find indices of all rows with this patient ID
        indices = df_ref[df_ref['Path'].str.contains(patient_id)].index.tolist()
        ids_to_remove.extend(indices)
    return ids_to_remove

# Get train data
train_df = pd.read_csv('../../../data_shares/purrlab/CheXpert/CheXpert-v1.0-small/train.csv', index_col=0)

# Apply the filters
# 1 - view is frontal
frontal_df = train_df[train_df['Frontal/Lateral'] == 'Frontal']
    
def filter_and_sample(df, sex, pneumothorax_status, n=7500):
    if pneumothorax_status == 1:
        filtered_df = df[(df['Sex'] == sex) & (df['Pneumothorax'] == 1)]
    else:
        # This will include females without pneumothorax (status not 1)
        filtered_df = df[(df['Sex'] == sex) & (df['Pneumothorax'] != 1)]
    
    # Sample if more than n entries are found
    if len(filtered_df) > n:
        return filtered_df.sample(n=n, random_state=5)
    else:
        return filtered_df

# 2 - sex is F and pneumothorax is 1, 7500 examples
female_p1 = filter_and_sample(frontal_df, 'Female', 1.0)

# 3 - sex is F and pneumothorax is -1, 7500 examples
female_pn1 = filter_and_sample(frontal_df, 'Female', 0)

# 2 - sex is M and pneumothorax is 1, 7500 examples
male_p1 = filter_and_sample(frontal_df, 'Male', 1.0)

# 5 - sex is M and pneumothorax is -1, 7500 examples
male_pn1 = filter_and_sample(frontal_df, 'Male', 0)

# Convert a data frame to a csv file
balanced_df = pd.concat([female_p1, female_pn1, male_p1, male_pn1])
balanced_dataset = balanced_df.sample(frac=1, random_state=42).reset_index(drop=False)
# balanced_df.to_csv('balanced2_subset.csv', index=False)

# Split into train, val and save to a csv file
train, remaining = train_test_split(balanced_dataset, train_size=0.7, random_state=0)
val, test = train_test_split(remaining, train_size=0.5, random_state=42)  # Split remaining evenly

idstoremove = get_patient_ids_to_remove(train, pd.concat([val, test]))
train = train.drop(idstoremove)
train.to_csv(f'{base_filename}_train.csv', index=False)

idstoremove = get_patient_ids_to_remove(test, val)
test = test.drop(idstoremove)

val.to_csv(f'{base_filename}_val.csv', index=False)
test.to_csv(f'{base_filename}_test.csv', index=False)