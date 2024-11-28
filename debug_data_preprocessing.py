########################################################################################################################
#
#   Script for debugging preprocessing the data for developing group-contribution based QSPRs
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
from src.data import construct_mg_data, remove_zero_one_sum_rows, filter_smiles
from src.splits import find_minimal_covering_smiles


# define the tag
prop_tag = 'Tc'

# data path
path_to_data = None
# select path based on property
if prop_tag in ['Tc','Pc', 'Vc', 'Omega']:
    path_to_data = 'data/external/critptops_mit.xlsx'
elif prop_tag == 'HCOM':
    pass

# read the data
df = pd.read_excel(path_to_data)
df = df[['SMILES', prop_tag]]

# drop SMILES that does not have a target value
df = df[~df[prop_tag].isna()]



# extract list of SMILES
lst_smiles = df['SMILES'].to_list()

# filter the smiles
filtered_smi, removed_smi = filter_smiles(lst_smiles)

# remove unwanted smiles
df_new = df[df['SMILES'].isin(filtered_smi)]


# reset the indices
df_new.reset_index(drop=True, inplace=True)

# save the smiles for icas fragment generation
file_paths = [
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg1.txt',
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg2.txt',
    './data/interim/'+prop_tag+'/'+prop_tag+'_mg3.txt']

# Check if file exists and write only if it doesn't
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Writing to {file_path}...")
        with open(file_path, 'w') as f:
            f.write('\n'.join(lst_smiles))  # Write the list of strings, one per line
    else:
        print(f"{file_path} already exists, skipping.")

#%%
# we need to combine the 3 txt files
def parse_chemical_file(file_path):
    """
    Parse a chemical data file that contains SMILES strings followed by a large table.

    Args:
        file_path (str): Path to the text file

    Returns:
        tuple: (pre_table_smiles, table_df)
            - pre_table_smiles: list of SMILES strings before the table
            - table_df: pandas DataFrame containing the table data
    """
    # Read all lines from file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find table start (line that begins with number column data)
    table_start_idx = None
    pre_table_smiles = []

    for idx, line in enumerate(lines):
        # Look for a line that matches the pattern: "No Smiles 1 2 3..."
        if line.strip().startswith('No') and 'Smiles' in line and '1' in line and '2' in line:
            table_start_idx = idx
            break
        else:
            # Store SMILES strings that appear before the table
            stripped_line = line.strip()
            if stripped_line:  # if line is not empty
                pre_table_smiles.append(stripped_line)

    if table_start_idx is None:
        raise ValueError("Could not find table start in file")

    # Extract table data starting from the header line
    table_lines = lines[table_start_idx:]

    # Split the header line to get column names
    header = table_lines[0].strip().split()

    # Create DataFrame from the rest of the lines
    data = []
    for line in table_lines[1:]:
        stripped_line = line.strip()
        if stripped_line:  # Skip empty lines
            # Split line by variable whitespace
            values = stripped_line.split()
            if len(values) >= len(header):  # Ensure line has enough values
                row_data = values[:len(header)]  # Take only as many values as there are headers
                data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)
    # Remove rows containing 'No'
    df = df[~df.isin(['No']).any(axis=1)]

    # # Convert numeric columns to float, handling 'No' values
    for col in df.columns[2:]:  # Skip 'No' and 'Smiles' columns
        df[col] = pd.to_numeric(df[col], errors='coerce')


    return pre_table_smiles, df
#%%
original_df = df_new
_, df_mg1 = parse_chemical_file('./data/interim/' + prop_tag + '/' + prop_tag +'_mg1.txt')
_, df_mg2 = parse_chemical_file('./data/interim/' + prop_tag + '/' + prop_tag + '_mg2.txt')
_, df_mg3 = parse_chemical_file('./data/interim/' + prop_tag + '/' + prop_tag + '_mg3.txt')
# # modify the column name og mg2 and mg3 to follow the order
df_mg2.columns = [str(int(col) + 220) if col.isdigit() else col for col in df_mg2.columns]
df_mg3.columns = [str(int(col) + 220 + 130) if col.isdigit() else col for col in df_mg3.columns]
# # need to remove those which first element is no -> did not fragment
#
# # # retrieve the original smiles (bug in icas that modifies smiles)
# fragmented_idx = [int(idx) - 1 for idx in df_mg1['No']]
# fragmented_smiles = df_new.loc[fragmented_idx, 'SMILES'].to_list()
# # # change the smiles in the fragmented dataframes
# df_mg1['Smiles'] = fragmented_smiles
# df_mg2['Smiles'] = fragmented_smiles
# df_mg3['Smiles'] = fragmented_smiles
# # # concatenate the three dataframes
# # #df_mg = concatenate_dataframes([df_mg1, df_mg2, df_mg3], key_columns=['Smiles'])
# df_mg = pd.concat([df_mg1, df_mg2.iloc[:, 2:], df_mg3.iloc[:, 2:]], axis=1)
# df_mg = df_mg.rename(columns={'Smiles':'SMILES'})
# # # find column tag

#df_mg.insert(loc=2, column='Const_Value', value=original_df[original_df['SMILES'].isin(fragmented_smiles)]['Const_Value'].values)
"O=NOCCC(C)C"
