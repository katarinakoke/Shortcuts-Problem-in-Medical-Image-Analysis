import os
import shutil
import numpy as np
import pandas as pd

# Get train data
train_df = pd.read_csv('../../../data_shares/purrlab/CheXpert/CheXpert-v1.0-small/train.csv', index_col=0)

# Apply the filters
# 1 - view is frontal
frontal_df = train_df[train_df['Frontal/Lateral'] == 'Frontal']

def filter_and_sample(df, sex, pneumothorax_status, n=7500):
    filtered_df = df[(df['Sex'] == sex) & (df['Pneumothorax'] == pneumothorax_status)]
    if len(filtered_df) > n:
        return filtered_df.sample(n=n, random_state=5)
    else:
        return filtered_df

# 2 - sex is F and pneumothorax is 1, 7500 examples
female_p1 = filter_and_sample(frontal_df, 'Female', 1.0)

# 3 - sex is F and pneumothorax is -1, 7500 examples
female_pn1 = filter_and_sample(frontal_df, 'Female', -1.0)

# 2 - sex is M and pneumothorax is 1, 7500 examples
male_p1 = filter_and_sample(frontal_df, 'Male', 1.0)

# 5 - sex is M and pneumothorax is -1, 7500 examples
male_pn1 = filter_and_sample(frontal_df, 'Male', -1.0)

# Convert a data frame to a csv file
balanced_df = pd.concat([female_p1, female_pn1, male_p1, male_pn1])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=False)
balanced_df.to_csv('balanced_subset.csv', index=False)

