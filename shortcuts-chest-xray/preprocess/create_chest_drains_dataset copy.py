import pandas as pd
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

# Getting the data and shuffling the rows
df = pd.read_csv('../chest_drains_dataset.csv', index_col=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=False)

# Creating a new column for the patient IDs
df['Patient_ID'] = df['Path'].apply(extract_patient_id)

# Splitting the dataset into training and remaining (test + validation) with patient-level separation
gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_inds, remaining_inds = next(gss.split(df, groups=df['Patient_ID']))
train_df = df.iloc[train_inds]
remaining_df = df.iloc[remaining_inds]

# Splitting the remaining into validation and test sets
gss_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
val_inds, test_inds = next(gss_val_test.split(remaining_df, groups=remaining_df['Patient_ID']))
val_df = remaining_df.iloc[val_inds]
test_df = remaining_df.iloc[test_inds]

# Removing the 'Patient_ID' since it is no longer needed for training
train_df = train_df.drop(columns=['Patient_ID'])
val_df = val_df.drop(columns=['Patient_ID'])
test_df = test_df.drop(columns=['Patient_ID'])

# Saving the datasets to CSV files
train_df.to_csv('../chest_drains_dataset_train.csv', index=False)
val_df.to_csv('../chest_drains_dataset_val.csv', index=False)
test_df.to_csv('../chest_drains_dataset_test.csv', index=False)