########################################################################################################################
#                                                                                                                      #
#    Collection of helper function and classes for data handling                                                       #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2024/12                                                                                                           #
#                                                                                                                      #
########################################################################################################################

##########################################################################################################
# Import packages and modules
##########################################################################################################
import pandas as pd
import numpy as np
from rdkit import Chem
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

##########################################################################################################
# Define functions and classes
##########################################################################################################

def parse_chemical_file(file_path):
    """
    Parse the file produced from "SMILES2Groups" produced by ProPred to organize the files and extract
    both smiles and groups in a dataframe. the dataframe produced contains the groups available in the file (either
    1st, 2nd, or 3rd)

    :param file_path: path to the file (.txt format)
    :return: dataframe with smiles and groups
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

def concatenate_dataframes(dfs, key_columns, how='outer'):
    """
    Concatenate multiple DataFrames based on multiple key columns.
    Drops 'No' column from all DataFrames except the first one.

    Args:
        dfs (list): List of pandas DataFrames to concatenate
        key_columns (str or list): Column name(s) to use as keys for concatenation
        how (str): Type of merge to perform ('outer', 'inner', 'left', 'right')

    Returns:
        pandas.DataFrame: Concatenated DataFrame
    """
    if not dfs:
        raise ValueError("No DataFrames provided")

    if len(dfs) < 2:
        raise ValueError("Need at least two DataFrames to concatenate")

    # Convert single key to list for consistent processing
    if isinstance(key_columns, str):
        key_columns = [key_columns]

    # Make copies to avoid modifying original DataFrames
    dfs = [df.copy() for df in dfs]

    # # Convert No column to integer in first DataFrame
    if 'No' in dfs[0].columns:
        dfs[0]['No'] = dfs[0]['No'].astype(int)

    # Drop 'No' column from all DataFrames except the first one
    for df in dfs[1:]:
        if 'No' in df.columns:
            df.drop('No', axis=1, inplace=True)

    # Start with the first DataFrame
    result = dfs[0]

    # Merge with remaining DataFrames
    for df in dfs[1:]:
        result = pd.merge(result, df, on=key_columns, how=how)

    # Sort by 'No' column
    result = result.sort_values('No', ignore_index=True)

    return result

def construct_mg_data(property_tag:str, original_df:pd.DataFrame):
    """
    function that constructs a dataframe with groups from MG
    :param property_tag:
    :param original_df:
    :return:
    """
    # load the dataframes
    _, df_mg1 = parse_chemical_file('./data/interim/' + property_tag + '/' + property_tag +'_mg1.txt')
    _, df_mg2 = parse_chemical_file('./data/interim/' + property_tag + '/' + property_tag + '_mg2.txt')
    _, df_mg3 = parse_chemical_file('./data/interim/' + property_tag + '/' + property_tag + '_mg3.txt')
    # modify the column name og mg2 and mg3 to follow the order
    df_mg2.columns = [str(int(col) + 220) if col.isdigit() else col for col in df_mg2.columns]
    df_mg3.columns = [str(int(col) + 220 + 130) if col.isdigit() else col for col in df_mg3.columns]
    # need to remove those which first element is no -> did not fragment

    # retrieve the original smiles (bug in icas that modifies smiles)
    fragmented_idx = [int(idx) - 1 for idx in df_mg1['No']]
    fragmented_smiles = original_df.loc[fragmented_idx, 'SMILES'].to_list()
    # change the smiles in the fragmented dataframes
    df_mg1['Smiles'] = fragmented_smiles
    df_mg2['Smiles'] = fragmented_smiles
    df_mg3['Smiles'] = fragmented_smiles
    # concatenate the three dataframes
    df_mg = pd.concat([df_mg1, df_mg2.iloc[:, 2:], df_mg3.iloc[:, 2:]], axis=1)
    df_mg = df_mg.rename(columns={'Smiles':'SMILES'})
    # find column tag
    try:
        df_mg.insert(loc=2, column='Const_Value',
                     value=original_df[original_df['SMILES'].isin(fragmented_smiles)]['Const_Value'].values)
    except (KeyError, ValueError) as e:
        df_mg.insert(loc=2, column=property_tag,
                     value=original_df[original_df['SMILES'].isin(fragmented_smiles)][property_tag].values)

    return df_mg

def remove_zero_one_sum_rows(df, columns_subset):
    """
    Remove rows where the sum of specified columns is zero or one.
    Preserves original index.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns_subset (list): List of column names to sum

    Returns:
        pd.DataFrame: DataFrame with zero/one-sum rows removed
    """
    # Calculate sum of specified columns
    row_sums = df[columns_subset].sum(axis=1)

    # Keep rows where sum is greater than 1
    df_filtered = df[row_sums > 1].copy()

    return df_filtered


def expand_subset(org_df, subset_df, target_size, random_seed=42):
    """
    Expand subset DataFrame by adding rows from original DataFrame until target size.

    Args:
        org_df (pd.DataFrame): Original DataFrame
        subset_df (pd.DataFrame): Subset DataFrame to expand
        target_size (int): Desired final size of subset
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Expanded subset DataFrame
    """
    # Input validation
    if target_size <= len(subset_df):
        print(f"Target size ({target_size}) is smaller than or equal to current subset size ({len(subset_df)})")
        return subset_df

    if target_size > len(org_df):
        print(f"Target size ({target_size}) is larger than original DataFrame size ({len(org_df)})")
        return subset_df

    # Set random seed
    np.random.seed(random_seed)

    # Get SMILES not already in subset
    current_smiles = set(subset_df['SMILES'])
    available_df = org_df[~org_df['SMILES'].isin(current_smiles)].copy()

    # Calculate how many new rows we need
    n_to_add = target_size - len(subset_df)

    if n_to_add > len(available_df):
        print(f"Warning: Can only add {len(available_df)} rows, not the requested {n_to_add}")
        n_to_add = len(available_df)

    # Randomly select rows to add
    rows_to_add = available_df.sample(n=n_to_add, random_state=random_seed)

    # Combine original subset with new rows
    expanded_df = pd.concat([subset_df, rows_to_add], ignore_index=False)
    # Add split column after smiles and reorder columns
    expanded_df = expanded_df.assign(label='train')
    cols = list(expanded_df.columns)
    label_idx = cols.index('label')
    smiles_idx = cols.index('SMILES')
    cols.pop(label_idx)
    cols.insert(smiles_idx + 2, 'label')
    expanded_df = expanded_df[cols]

    # Get remaining data (not used in expanded subset)
    used_smiles = set(expanded_df['SMILES'])
    remaining_df = org_df[~org_df['SMILES'].isin(used_smiles)].copy()

    # Split remaining data into validation and test sets
    remaining_size = len(remaining_df)
    val_size = remaining_size // 2  # Split remaining data roughly in half

    # Randomly shuffle and split
    val_df = remaining_df.sample(n=val_size, random_state=random_seed).copy()
    test_df = remaining_df[~remaining_df.index.isin(val_df.index)].copy()

    # Add split column for validation and test sets and reorder columns
    val_df = val_df.assign(label='val')
    test_df = test_df.assign(label='test')

    # Reorder columns for val and test sets
    cols = list(val_df.columns)
    label_idx = cols.index('label')
    smiles_idx = cols.index('SMILES')
    cols.pop(label_idx)
    cols.insert(smiles_idx + 2, 'label')
    val_df = val_df[cols]
    test_df = test_df[cols]

    return expanded_df, val_df, test_df


def filter_smiles(smiles_list: List[str], prohibited_smiles: Optional[List[str]] = None) -> Tuple[
    List[str], Dict[str, List[str]]]:
    """
    Filter SMILES strings based on multiple criteria and track rejections.

    Args:
        smiles_list (List[str]): List of SMILES strings to filter
        prohibited_smiles (Optional[List[str]]): List of prohibited SMILES strings

    Returns:
        Tuple[List[str], Dict[str, List[str]]]:
            - List of accepted SMILES strings
            - Dictionary mapping rejection reasons to lists of rejected SMILES
    """
    if prohibited_smiles is None:
        prohibited_smiles = ['C', 'CC', '[HH]', 'O=C=O', 'C(F)(F)(F)S(F)(F)(F)(F)F']

    filtered_smiles = []
    rejected_smiles = defaultdict(list)

    for smiles in smiles_list:
        # Check if in prohibited list
        if smiles in prohibited_smiles:
            rejected_smiles["prohibited_list"].append(smiles)
            continue

        # Try to create RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected_smiles["invalid_smiles"].append(smiles)
            continue

        # Check number of heavy atoms
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        if num_heavy_atoms <= 1:
            rejected_smiles["single_heavy_atom"].append(smiles)
            continue

        # Check for presence of carbon
        has_carbon = False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon atomic number
                has_carbon = True
                break

        if not has_carbon:
            rejected_smiles["no_carbon"].append(smiles)
            continue

        filtered_smiles.append(smiles)

    return filtered_smiles, dict(rejected_smiles)

def move_column(df, column_name, position=3):
    # Get the column, drop it from the DataFrame, and reinsert at desired position
    col = df.pop(column_name)
    df.insert(position, column_name, col)
    return df