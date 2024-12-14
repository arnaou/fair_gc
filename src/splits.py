from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem
import numpy as np
from typing import List, Tuple, Dict
import random
import pandas as pd


def find_minimal_covering_smiles(df):
    """
    Find a set of SMILES that covers all non-zero features in the DataFrame,
    preferring SMILES that cover fewer features at each step.

    Args:
        df (pandas.DataFrame): Input DataFrame with 'smiles' column and feature columns

    Returns:
        tuple: (list of selected SMILES, dict of feature coverage details)
    """
    # Get feature columns (all numeric columns from 1 to 424)
    feature_cols = [str(i) for i in range(1, 425)]

    # Create a binary matrix where 1 indicates presence of feature
    binary_matrix = (df[feature_cols] != 0).astype(int)

    # Keep track of uncovered features and selected SMILES
    uncovered_features = set(col for col in feature_cols
                             if binary_matrix[col].sum() > 0)  # only include features that exist
    selected_smiles = []
    coverage_details = {}
    used_indices = set()

    while uncovered_features:
        # Count how many uncovered features each SMILES would add
        feature_counts = []
        for idx, row in binary_matrix.iterrows():
            if idx not in used_indices:  # Only consider unused SMILES
                # Count only features that are both present (1) and still uncovered
                new_features = sum(1 for col in uncovered_features
                                   if row[col] == 1)
                if new_features > 0:  # Only include if it covers at least one uncovered feature
                    feature_counts.append((idx, new_features))

        if not feature_counts:
            print("Warning: Some features could not be covered!")
            break

        # Find SMILES that covers least uncovered features (but at least one)
        best_idx, num_new_features = min(feature_counts, key=lambda x: x[1])

        # Add this SMILES to our solution
        selected_smile = df.loc[best_idx, 'SMILES']
        selected_smiles.append(selected_smile)
        used_indices.add(best_idx)

        # Record which features this SMILES covers
        covered_features = [col for col in uncovered_features
                            if binary_matrix.loc[best_idx, col] == 1]
        coverage_details[selected_smile] = {
            'covers_features': covered_features,
            'values': {col: df.loc[best_idx, col] for col in covered_features}
        }

        # Update uncovered features
        uncovered_features -= set(covered_features)

    return selected_smiles, coverage_details

def find_nonzero_columns(df, exclude_cols):
    """
    Find columns in DataFrame that are not all zeros.
    Excludes non-numeric columns like 'smiles' and 'split'.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        list: Column names that have at least one non-zero value
        :param exclude_cols:
    """
    # Exclude non-numeric columns
    numeric_cols = [col for col in df.columns if col not in exclude_cols]

    # Find columns with any non-zero value
    nonzero_cols = []
    for col in numeric_cols:
        if (df[col] != 0).any():
            nonzero_cols.append(col)

    return nonzero_cols


def split_indices(indices_list):
    """
    Split list of string indices into three groups based on boundaries.

    Args:
        indices_list (list): List of string indices

    Returns:
        tuple: (below_lower, between_bounds, above_upper)
    """
    # Convert to integers for comparison
    indices = [int(x) for x in indices_list]
    lower = 220
    upper = 350

    # Split into groups
    idx_G1 = [str(x) for x in indices if x <= lower]
    idx_G2 = [str(x) for x in indices if lower < x <= upper]
    idx_G3 = [str(x) for x in indices if x > upper]

    return idx_G1, idx_G2, idx_G3

def split_molecules_by_clusters(
        smiles_list: List[str],
        split_fractions: List[float],
        cutoff: float = 0.65,
        random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split molecules into train/validation/test sets by sampling within each cluster
    according to split fractions, using the largest cluster for final adjustments.

    Args:
        smiles_list: List of SMILES strings
        split_fractions: List of fractions [train, validation, test] that sum to 1
        cutoff: Tanimoto similarity cutoff for Butina clustering
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_smiles, val_smiles, test_smiles)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    assert len(split_fractions) == 3, "Must provide 3 split fractions"
    assert abs(sum(split_fractions) - 1.0) < 1e-6, "Split fractions must sum to 1"

    # Calculate target sizes for the entire dataset
    total_size = len(smiles_list)
    target_sizes = [int(round(total_size * frac)) for frac in split_fractions]
    target_sizes[-1] = total_size - sum(target_sizes[:-1])  # Adjust last split

    # Convert SMILES to molecules and compute fingerprints
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]

    # Calculate distance matrix
    n = len(fps)
    dists = []
    for i in range(n):
        for j in range(i):
            dist = 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(dist)

    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    cluster_lists = [list(cluster) for cluster in clusters]

    # Sort clusters by size (smallest first)
    cluster_lists.sort(key=len)

    # Initialize output sets
    train_indices = set()
    val_indices = set()
    test_indices = set()

    def split_cluster(cluster: List[int], fractions: List[float]) -> Tuple[List[int], List[int], List[int]]:
        """Split a single cluster according to given fractions."""
        size = len(cluster)
        indices = list(cluster)
        random.shuffle(indices)

        # Calculate split sizes for this cluster
        split_sizes = [int(round(size * frac)) for frac in fractions]

        # Adjust last split size to ensure all molecules are assigned
        remaining = size - sum(split_sizes[:-1])
        split_sizes[-1] = remaining

        # Split the indices
        current_idx = 0
        splits = []
        for split_size in split_sizes:
            splits.append(indices[current_idx:current_idx + split_size])
            current_idx += split_size

        return splits[0], splits[1], splits[2]

    # Handle all clusters except the largest one
    largest_cluster = cluster_lists.pop()
    current_splits = [len(train_indices), len(val_indices), len(test_indices)]

    # Process all smaller clusters
    for cluster in cluster_lists:
        if len(cluster) <= 3:
            if len(cluster) == 1:
                train_indices.update(cluster)
            elif len(cluster) == 2:
                train_indices.add(cluster[0])
                val_indices.add(cluster[1])
            else:  # size == 3
                train_indices.add(cluster[0])
                val_indices.add(cluster[1])
                test_indices.add(cluster[2])
        else:
            # Split according to target fractions
            train_split, val_split, test_split = split_cluster(cluster, split_fractions)
            train_indices.update(train_split)
            val_indices.update(val_split)
            test_indices.update(test_split)

    # Calculate how many samples we need from the largest cluster for each split
    remaining_train = target_sizes[0] - len(train_indices)
    remaining_val = target_sizes[1] - len(val_indices)
    remaining_test = len(largest_cluster) - remaining_train - remaining_val

    # Shuffle largest cluster
    random.shuffle(largest_cluster)


    # Assign from the largest cluster to meet targets exactly
    current_idx = 0

    # Add to training set
    train_indices.update(largest_cluster[current_idx:current_idx + remaining_train])
    current_idx += remaining_train

    # Add to validation set
    val_indices.update(largest_cluster[current_idx:current_idx + remaining_val])
    current_idx += remaining_val

    # Add remaining to test set
    test_indices.update(largest_cluster[current_idx:])

    # Convert back to SMILES
    train_smiles = [smiles_list[i] for i in sorted(train_indices)]
    val_smiles = [smiles_list[i] for i in sorted(val_indices)]
    test_smiles = [smiles_list[i] for i in sorted(test_indices)]

    # Print split statistics
    print("Target split fractions:", split_fractions)
    print("Actual split fractions:", [
        len(train_smiles) / total_size,
        len(val_smiles) / total_size,
        len(test_smiles) / total_size
    ])

    return train_smiles, val_smiles, test_smiles


def expand_subset(org_df, subset_df, target_size, method="random", butina_cutoff=0.6, random_seed=42):
    """
    Expand subset DataFrame by adding rows from original DataFrame until target size.

    Args:
        org_df (pd.DataFrame): Original DataFrame
        subset_df (pd.DataFrame): Subset DataFrame to expand
        target_size (int): Desired final size of subset
        method (str): either "random" or "butina"
        butina_cutoff: float 0-1
        random_seed (int): Random seed for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Expanded train, validation, and test DataFrames
    """
    # Input validation
    if target_size <= len(subset_df):
        print(f"Target size ({target_size}) is smaller than or equal to current subset size ({len(subset_df)})")
        return subset_df, pd.DataFrame(), pd.DataFrame()

    if target_size > len(org_df):
        print(f"Target size ({target_size}) is larger than original DataFrame size ({len(org_df)})")
        return subset_df, pd.DataFrame(), pd.DataFrame()

    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Get SMILES not already in subset
    current_smiles = set(subset_df['SMILES'])
    available_df = org_df[~org_df['SMILES'].isin(current_smiles)].copy()

    # Calculate how many new rows we need
    n_to_add = target_size - len(subset_df)

    if n_to_add > len(available_df):
        print(f"Warning: Can only add {len(available_df)} rows, not the requested {n_to_add}")
        n_to_add = len(available_df)

    if method == 'random':
        # Randomly select rows to add
        rows_to_add = available_df.sample(n=n_to_add, random_state=random_seed)
        # Combine original subset with new rows
        expanded_df = pd.concat([subset_df, rows_to_add], ignore_index=False)

        # Get remaining data (not used in expanded subset)
        used_smiles = set(expanded_df['SMILES'])
        remaining_df = org_df[~org_df['SMILES'].isin(used_smiles)].copy()

        # Split remaining data into validation and test sets
        remaining_size = len(remaining_df)
        val_size = remaining_size // 2  # Split remaining data roughly in half
        test_size = remaining_size - val_size  # Ensure we account for odd numbers

        # Split remaining data
        val_df = remaining_df.sample(n=val_size, random_state=random_seed).copy()
        test_df = remaining_df[~remaining_df.index.isin(val_df.index)].copy()

    elif method == 'butina':
        # Calculate subset sizes
        n_available = len(available_df)
        n_val = (n_available - n_to_add) // 2
        n_test = n_available - n_to_add - n_val  # Ensure total equals n_available

        # Calculate the ratios
        ratios = [
            n_to_add / n_available,
            n_val / n_available,
            1 - (n_to_add + n_val) / n_available
        ]

        # Get SMILES list
        smi_list = available_df['SMILES'].tolist()

        # Perform Butina-based splitting
        train_smiles, val_smiles, test_smiles = split_molecules_by_clusters(
            smiles_list=smi_list,
            split_fractions=ratios,
            cutoff=butina_cutoff,
            random_seed=random_seed
        )

        # Convert back to DataFrames, ensuring exact sizes
        rows_to_add = available_df[available_df['SMILES'].isin(train_smiles)].copy()
        # if len(rows_to_add) > n_to_add:
        #     rows_to_add = rows_to_add.sample(n=n_to_add, random_state=random_seed)

        val_df = available_df[available_df['SMILES'].isin(val_smiles)].copy()
        # if len(val_df) > n_val:
        #     val_df = val_df.sample(n=n_val, random_state=random_seed)

        test_df = available_df[available_df['SMILES'].isin(test_smiles)].copy()
        # if len(test_df) > n_test:
        #     test_df = test_df.sample(n=n_test, random_state=random_seed)

        # Combine original subset with new rows
        train_df = pd.concat([subset_df, rows_to_add], ignore_index=False)

    else:
        raise ValueError(f"Unknown method: {method}. Use either 'random' or 'butina'")

    # Add label column after SMILES and reorder columns
    train_df = train_df.assign(label='train')
    val_df = val_df.assign(label='val')
    test_df = test_df.assign(label='test')

    # Reorder columns
    for df in [train_df, val_df, test_df]:
        cols = list(df.columns)
        label_idx = cols.index('label')
        smiles_idx = cols.index('SMILES')
        cols.pop(label_idx)
        cols.insert(smiles_idx + 2, 'label')
        df = df[cols]

    # final check
    # if method == 'butina':
    #     check_split_fractions(
    #         len(expanded_df),
    #         len(val_df),
    #         len(test_df),
    #         ratios,  # for butina method
    #         tolerance=0.05  # adjust this value as needed
    #     )
    # Check for duplicate SMILES
    # Remove duplicate SMILES

    # expanded_df, val_df, test_df = remove_duplicate_smiles(expanded_df, val_df, test_df)
    # check_duplicate_smiles(expanded_df, val_df, test_df)

    # Verification before return
    total_samples = len(train_df) + len(val_df) + len(test_df)
    if total_samples != len(org_df):
        print(f"WARNING: Total samples {total_samples} does not match original dataset size {len(org_df)}")
        print(f"Method: {method}")
        print(f"Training size: {len(train_df)}")
        print(f"Validation size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")
        print(f"Original size: {len(org_df)}")

    return train_df, val_df, test_df


def check_split_fractions(train_size, val_size, test_size, target_ratios, tolerance=0.05):
    """
    Check if actual split fractions match target ratios within tolerance.

    Args:
        train_size (int): Size of training set
        val_size (int): Size of validation set
        test_size (int): Size of test set
        target_ratios (List[float]): Target split ratios [train, val, test]
        tolerance (float): Acceptable deviation from target ratios

    Returns:
        None (prints warning if fractions deviate too much)
    """
    total = train_size + val_size + test_size
    actual_ratios = [train_size / total, val_size / total, test_size / total]

    max_deviation = max(abs(actual - target) for actual, target in zip(actual_ratios, target_ratios))
    if max_deviation > tolerance:
        print(f"WARNING: Split fractions deviate from target ratios by {max_deviation:.3f}")
        print(f"Target ratios: {[f'{r:.3f}' for r in target_ratios]}")
        print(f"Actual ratios: {[f'{r:.3f}' for r in actual_ratios]}")


def check_duplicate_smiles(train_df, val_df, test_df):
    """
    Check for duplicate SMILES across train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training set DataFrame
        val_df (pd.DataFrame): Validation set DataFrame
        test_df (pd.DataFrame): Test set DataFrame

    Returns:
        None (prints warning if duplicates are found)
    """
    train_smiles = set(train_df['SMILES'])
    val_smiles = set(val_df['SMILES'])
    test_smiles = set(test_df['SMILES'])

    train_val = train_smiles.intersection(val_smiles)
    train_test = train_smiles.intersection(test_smiles)
    val_test = val_smiles.intersection(test_smiles)

    if train_val or train_test or val_test:
        print("\nWARNING: Duplicate SMILES detected!")
        if train_val:
            print(f"Found {len(train_val)} duplicates between train and validation sets")
        if train_test:
            print(f"Found {len(train_test)} duplicates between train and test sets")
        if val_test:
            print(f"Found {len(val_test)} duplicates between validation and test sets")


def remove_duplicate_smiles(train_df, val_df, test_df):
    """
    Remove any duplicate SMILES across train, validation, and test sets.
    If a SMILES appears in multiple sets, keep it only in the training set.

    Args:
        train_df (pd.DataFrame): Training set DataFrame
        val_df (pd.DataFrame): Validation set DataFrame
        test_df (pd.DataFrame): Test set DataFrame

    Returns:
        tuple: (train_df, val_df, test_df) with duplicates removed
    """
    train_smiles = set(train_df['SMILES'])

    # Remove any SMILES that also appear in training set
    val_df = val_df[~val_df['SMILES'].isin(train_smiles)].copy()
    test_df = test_df[~test_df['SMILES'].isin(train_smiles)].copy()

    # Remove any SMILES that appear in both val and test
    val_smiles = set(val_df['SMILES'])
    test_df = test_df[~test_df['SMILES'].isin(val_smiles)].copy()

    return train_df, val_df, test_df