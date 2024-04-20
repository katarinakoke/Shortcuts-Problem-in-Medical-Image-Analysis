import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def extract_patient_id(path):
    '''Takes the path to the image and returns the part containing the patient ID.'''
    parts = path.split('/')
    for part in parts:
        if part.startswith('patient'):
            return part
    return None

def get_patient_ids_to_remove(df_ref, df_subset):
    '''Takes two datasets to check for matching patients in both datasets. 
        It returns a list of patient IDs from the subset that can be found in the referenced dataset.'''
    subset_patient_ids = df_subset['Path'].apply(extract_patient_id).unique()
    ids_to_remove = []
    for patient_id in subset_patient_ids:
        # Find indices of all rows with this patient ID
        indices = df_ref[df_ref['Path'].str.contains(patient_id)].index.tolist()
        ids_to_remove.extend(indices)
    return ids_to_remove

# Getting the data and the chest drain annotations
ChestX_ray14 = pd.read_csv('../../../../data_shares/purrlab_students/ChestX-ray14/Data_Entry_2017.csv', index_col=0)
drain_annotations = pd.read_csv('NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv', index_col=0)
drain_annotations['Drain'] = drain_annotations['Drain'].map({0: -1, 1: 1})

# Combining the datasets with an anti-join
df = pd.merge(ChestX_ray14, drain_annotations, on='Image Index', how='outer', indicator=True)
df = df[(df._merge=='left_only')].drop('_merge', axis=1)

# Adding a new column for Pneumothorax labels only
# df['Pneumothorax'] = df['Finding Labels'].str.contains('Pneumothorax').astype(int) -- labels as 0 znd 1
df['Pneumothorax'] = np.where(df['Finding Labels'].str.contains('Pneumothorax'), 1, -1)

# Dropping unecessary columns
df = df.drop(['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11', 'Drain_a1', 'Drain_Location_a1', 'Drain_a2', 'Drain_Location_a2'], axis = 1)

# Remove rows where the value of Pneumothorax is 1
df = df[df['Pneumothorax'] != 1]

# Saving the csv file
df.to_csv('../chest_drains_to_annotate.csv', index=False)

# Splitting the dataset into training and remaining (test + validation) with patient-level separation
gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_inds, remaining_inds = next(gss.split(df, groups=df['Patient ID']))
train_df = df.iloc[train_inds]
remaining_df = df.iloc[remaining_inds]

# Splitting the remaining into validation and test sets
gss_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
val_inds, test_inds = next(gss_val_test.split(remaining_df, groups=remaining_df['Patient ID']))
val_df = remaining_df.iloc[val_inds]
test_df = remaining_df.iloc[test_inds]

# Saving the datasets to CSV files
# train_df.to_csv('../chest_drains_dataset_train.csv', index=False)
# val_df.to_csv('../chest_drains_dataset_val.csv', index=False)
# test_df.to_csv('../chest_drains_dataset_test.csv', index=False)