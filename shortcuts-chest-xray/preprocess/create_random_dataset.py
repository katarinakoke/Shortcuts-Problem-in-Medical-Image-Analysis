import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

base_filename = 'random_dataset' 

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

# 2 - take 30k random samples
def sample_dataset(df, sample_size, random_state=None):
    return df.sample(n=sample_size, random_state=random_state, replace=False)

random_dataset = sample_dataset(frontal_df, 30000, random_state=42).reset_index(drop=False)

# Split into train, val and save to a csv file
train, remaining = train_test_split(random_dataset, train_size=0.7, random_state=0)
val, test = train_test_split(remaining, train_size=0.5, random_state=42)  # Split remaining evenly

idstoremove = get_patient_ids_to_remove(train, pd.concat([val, test]))
train = train.drop(idstoremove)
train.to_csv(f'{base_filename}_train.csv', index=False)

idstoremove = get_patient_ids_to_remove(test, val)
test = test.drop(idstoremove)

val.to_csv(f'../{base_filename}_val.csv', index=False)
test.to_csv(f'../{base_filename}_test.csv', index=False)

