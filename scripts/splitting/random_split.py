########################################################################################################################
#
#   Script for preprocessing the data for developing group-contribution based QSPRs
#
#
#
#
#   Adem R.N. Aouichaoui, arnaou@kt.dtu.dk
#   24-11-2024
########################################################################################################################

# import packages
import pandas as pd
import os
from src.data import construct_mg_data, remove_zero_one_sum_rows, filter_smiles, move_column
from sklearn.model_selection import train_test_split
import numpy as np

# define the tag
prop_tag = 'Omega'
split_type = 'random'

# data path
path_to_data = None
# select path based on property
if prop_tag in ['Tc','Pc', 'Vc', 'Omega']:
    path_to_data = '../data/external/critptops_mit.xlsx'
elif prop_tag == 'HCOM':
    pass

# read the data
df0 = pd.read_excel(path_to_data)
df0 = df0[['SMILES', prop_tag]]

# drop SMILES that does not have a target value
df0 = df0[~df0[prop_tag].isna()]

# extract list of SMILES
all_smiles = df0['SMILES'].to_list()

# clean the SMILES
filtered_smi, removed_smi = filter_smiles(smiles_list= all_smiles)

# clean and update the df
df = df0[df0['SMILES'].isin(filtered_smi)]
df.reset_index(drop=True, inplace=True)

# # save the smiles for icas fragment generation
file_paths = [
    '../data/interim/'+prop_tag+'/'+prop_tag+'_mg1.txt',
    '../data/interim/'+prop_tag+'/'+prop_tag+'_mg2.txt',
    '../data/interim/'+prop_tag+'/'+prop_tag+'_mg3.txt']

# Check if file exists and write only if it doesn't
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Writing to {file_path}...")
        with open(file_path, 'w') as f:
            f.write('\n'.join(filtered_smi))  # Write the list of strings, one per line
    else:
        print(f"{file_path} already exists, skipping.")
#%%
# construct the occurrence matrix
df_mg = construct_mg_data(prop_tag, df)
# step0: remove all compounds where the sum of groups is 0 --> could not be segmented
columns = [str(i) for i in range(1, 425)]
df_mg = remove_zero_one_sum_rows(df_mg, columns)
df_mg.reset_index(drop=True, inplace=True)
# step1: define the split ratio
test_size = 0.15
val_size = 0.15
train_size = 0.7
# Set seed for reproducibility
seeds = [7, 13, 19, 9, 48, 1, 8, 55, 100]   # You can choose any integer value
# name of the excel_file
output_excel_file = '../data/processed/'+prop_tag+'/'+prop_tag+'_'+split_type+'_processed.xlsx'
with pd.ExcelWriter(output_excel_file) as writer:
    for seed in seeds:
        print(f"Processing split for seed: {seed}")
        np.random.seed(seed)  # Set the seed for the current split

        # Perform the train-validation-test split
        train_val_df, test_df = train_test_split(df_mg, test_size=test_size, random_state=seed)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size/(train_size + val_size), random_state=seed)

        # Add the label column
        train_df['label'] = 'train'
        val_df['label'] = 'val'
        test_df['label'] = 'test'

        # Concatenate the DataFrames
        df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Move the 'label' column
        df_final = move_column(df_final, 'label', 3)

        # Save to a separate sheet with the specified naming convention
        sheet_name = f'{prop_tag}_random_{seed}'
        df_final.to_excel(writer, sheet_name=sheet_name, index=False)

        # Verify the split for the current seed (optional)
        n = len(df_mg)
        print(f"  Original DataFrame: {n} rows")
        print(f"  Train set: {len(train_df)} rows ({len(train_df)/n:.2%})")
        print(f"  Validation set: {len(val_df)} rows ({len(val_df)/n:.2%})")
        print(f"  Test set: {len(test_df)} rows ({len(test_df)/n:.2%})")

print(f"\nData splits saved to different sheets in '{output_excel_file}'")
