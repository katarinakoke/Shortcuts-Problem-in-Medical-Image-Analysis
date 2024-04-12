import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

base_filename = 'biased_dataset' 

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
train_df = pd.read_csv('../../../../data_shares/purrlab/CheXpert/CheXpert-v1.0-small/train.csv', index_col=0)

# Apply the filters
# 1 - view is frontal
frontal_df = train_df[train_df['Frontal/Lateral'] == 'Frontal']
    
def filter_and_sample(df, sex, pneumothorax_status, n):
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

# 2 - sex is F and pneumothorax is 1, 3000 examples
female_p1 = filter_and_sample(frontal_df, 'Female', 1.0, n=3000)

# 3 - sex is F and pneumothorax is -1, 3000 examples
female_pn1 = filter_and_sample(frontal_df, 'Female', 0, n=3000)

# 2 - sex is M and pneumothorax is 1, 12000 examples
male_p1 = filter_and_sample(frontal_df, 'Male', 1.0, n=12000)

# 5 - sex is M and pneumothorax is -1, 12000 examples
male_pn1 = filter_and_sample(frontal_df, 'Male', 0, n=12000)

# Convert a data frame to a csv file
biased_df = pd.concat([female_p1, female_pn1, male_p1, male_pn1])
biased_df = biased_df.sample(frac=1, random_state=42).reset_index(drop=False)
# biased_df.to_csv('biased_subset.csv', index=False)

# Let's assume 'biased_df' has already been prepared up to the shuffling stage and includes 'Path'
# First, extract patient IDs and prepare the stratification group
biased_df['Patient_ID'] = biased_df['Path'].apply(extract_patient_id)
biased_df['Stratify_Group'] = biased_df['Sex'].astype(str) + "_" + biased_df['Pneumothorax'].astype(str)

# Initialize the GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)

# Splitting the dataset into training and remaining (test + validation) ensuring patient-level separation
train_inds, remaining_inds = next(gss.split(biased_df, groups=biased_df['Patient_ID']))
train_df = biased_df.iloc[train_inds]
remaining_df = biased_df.iloc[remaining_inds]

# Further split the remaining into validation and test sets
gss_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
val_inds, test_inds = next(gss_val_test.split(remaining_df, groups=remaining_df['Patient_ID']))
val_df = remaining_df.iloc[val_inds]
test_df = remaining_df.iloc[test_inds]

# Optionally remove the 'Patient_ID' and 'Stratify_Group' if they are no longer needed for training
train_df = train_df.drop(columns=['Patient_ID', 'Stratify_Group'])
val_df = val_df.drop(columns=['Patient_ID', 'Stratify_Group'])
test_df = test_df.drop(columns=['Patient_ID', 'Stratify_Group'])

# Save the datasets to CSV files
train_df.to_csv('../biased_dataset_train.csv', index=False)
val_df.to_csv('../biased_dataset_val.csv', index=False)
test_df.to_csv('../biased_dataset_test.csv', index=False)