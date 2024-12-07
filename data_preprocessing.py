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
from src.splits import find_minimal_covering_smiles, expand_subset, find_nonzero_columns, split_indices

# define the tag
prop_tag = 'Vc'

# data path
path_to_data = None
# select path based on property
if prop_tag in ['Tc','Pc', 'Vc', 'Omega']:
    path_to_data = 'data/external/critptops_mit.xlsx'
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
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg1.txt',
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg2.txt',
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg3.txt']

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
# step 1: ensure that for each dataset, all available groups are available in a subset that is the training set
minimal_set, coverage = find_minimal_covering_smiles(df_mg)
df_train = df_mg[df_mg['SMILES'].isin(minimal_set)]
# make a copy of yhe minimal training set
df_train_min = df_train.copy()
# step 2: check the percentage of current training data
ratio_train = df_train.shape[0]/df_mg.shape[0]

# step 3: fill up randomly until a quota is reached
target_ratio = 0.70

# add to the train set and devide into validation and test set
n_train_to_add = int(target_ratio*df_mg.shape[0])

# perform the splitting
split_type = 'butina'
df_train, df_val, df_test = expand_subset(df_mg, df_train, n_train_to_add, method=split_type, butina_cutoff=0.6, random_seed=42)


#%%

df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)

df_final = move_column(df_final, 'label', 3)
df_final['required'] = [False for i in range(df_final.shape[0])]
df_final.loc[df_final['SMILES'].isin(df_train_min['SMILES']),'required'] = True

df_final = move_column(df_final, 'required', 4)


df_final.to_excel('data/processed/'+prop_tag+'/'+prop_tag+'_processed.xlsx', index=False)

# # retrieve indices of available groups
# idx_avail = find_nonzero_columns(df_train, ['SMILES', 'Const_Value', 'label', 'No'])
# idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)


