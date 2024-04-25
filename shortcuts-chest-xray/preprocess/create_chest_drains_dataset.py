import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# def extract_patient_id(path):
#     '''Takes the path to the image and returns the part containing the patient ID.'''
#     parts = path.split('/')
#     for part in parts:
#         if part.startswith('patient'):
#             return part
#     return None


# def extract_image_id(path):
#     '''Takes the path to the image and returns the part containing the image ID.'''
#     parts = path.split('/')
#     for part in parts:
#         if part.endswith('png'):
#             return part
#     return None

# def get_patient_ids_to_remove(df_ref, df_subset):
#     '''Takes two datasets to check for matching patients in both datasets. 
#         It returns a list of patient IDs from the subset that can be found in the referenced dataset.'''
#     subset_patient_ids = df_subset['Path'].apply(extract_patient_id).unique()
#     ids_to_remove = []
#     for patient_id in subset_patient_ids:
#         # Find indices of all rows with this patient ID
#         indices = df_ref[df_ref['Path'].str.contains(patient_id)].index.tolist()
#         ids_to_remove.extend(indices)
#     return ids_to_remove

# # Getting the data and shuffling the rows
# df_no_pneumothorax = pd.read_csv('../chest_drains_annotations_without_pneumothorax.csv')
# df_pneumothorax = pd.read_csv('NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv', index_col=0)
# df_original = pd.read_csv("../../../../data_shares/purrlab_students/ChestX-ray14/Data_Entry_2017.csv")

# # Creating a new column for the image IDs
# df_no_pneumothorax['Image Index'] = df_no_pneumothorax['image'].apply(extract_image_id) 

# df  = df_original.merge(df_no_pneumothorax,on='Image Index', how='outer')
# df  = df.merge(df_pneumothorax,on='Image Index', how='outer')

# df['Drain_x'] = df['Drain_x'].map({'Chest Drain': 1, 'No Chest Drain': 0})




# df['Drain']= df['Drain_x'].combine_first(df['Drain_y']) 


# df = df.drop(['Drain_x','Drain_y','OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11', 'Drain_a1', 'Drain_Location_a1', 'Drain_a2', 'Drain_Location_a2', 'annotation_id','annotator','created_at','id','image','lead_time','updated_at'], axis=1)


# df = df.dropna(subset=['Drain'],axis=0)
# df['Drain'] = df['Drain'].map({2: 1, 1:1, 0:0})

# df = df.sort_values(by ='Image Index')
# df = df.drop_duplicates(subset='Image Index', keep='first')




df = pd.read_csv("datasets/chest_drains_annotations.csv")
#df.to_csv('../chest_drains_annotations_no_duplicates.csv', index=False)

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

# Removing the 'Patient_ID' since it is no longer needed for training
train_df = train_df.drop(columns=['Patient ID'])
val_df = val_df.drop(columns=['Patient ID'])
test_df = test_df.drop(columns=['Patient ID'])

# Saving the datasets to CSV files
train_df.to_csv('datasets/chest_drains_dataset_train.csv', index=False)
val_df.to_csv('datasets/chest_drains_dataset_val.csv', index=False)
test_df.to_csv('datasets/chest_drains_dataset_test.csv', index=False)