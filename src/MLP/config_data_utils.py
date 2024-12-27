import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from sklearn.preprocessing import StandardScaler


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data path
# DATA_PATH = '/zhome/32/3/215274/jinlia/project/data/hcomb_data.xlsx'

def calc_MARE(ym, yp):
    """Calculate Modified Average Relative Error"""
    RAE = []
    pstd = np.std(ym)
    for i in range(0, len(ym)):
        if 0.1 >= ym[i] >= -0.1:
            RAE.append(abs(ym[i]-yp[i])/pstd*100)
        else:
            RAE.append(abs(ym[i]-yp[i])/abs(ym[i]+0.000001)*100)
    return np.mean(RAE)

def calculate_metrics(y_target, y_pred, n_params=None):
    """Calculate various performance metrics"""
    r2 = r2_score(y_target, y_pred)
    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
    mse = mean_squared_error(y_target, y_pred)
    mare = calc_MARE(y_target, y_pred)
    mae = mean_absolute_error(y_target, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mse': mse, 'mare': mare, 'mae': mae}

# def prepare_data():
#     """Load and prepare data from Excel file"""
#     try:
#         df = pd.read_excel(DATA_PATH)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure the file exists.")
    
#     # Extract features and target

#     # X = df.iloc[:, 4:].values  # All columns from 5th onwards are features

#     # Get all feature columns
#     X = df.iloc[:, 4:]
#     # Remove columns that are all zeros
#     X = X.loc[:, (X != 0).any(axis=0)]
#     X = X.values  # Convert to numpy array
#     print(X.shape)

#     y = df['Const_Value'].values
#     labels = df['label'].values
    
#     # Split based on predefined labels
#     X_train = X[labels == 'train']
#     X_val = X[labels == 'val']
#     X_test = X[labels == 'test']
    
#     y_train = y[labels == 'train']
#     y_val = y[labels == 'val']
#     y_test = y[labels == 'test']
    
#     return X_train, X_val, X_test, y_train, y_val, y_test

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_predictions(df, model, device, dataset_name='hcomb', y_name='Const_Value', 
                    train_label='label', smiles='SMILES', other_columns=['No'], scaler=None,additional_data=None):
    """Save predictions while maintaining original data order"""
    model.eval()
    
    # Prepare features in the same way as training
    feature_df = df.drop(columns=[smiles] + other_columns + [y_name, train_label])
    X = torch.FloatTensor(feature_df.values).to(device)
    X = X[:, (X != 0).any(axis=0)]
    # Get predictions in original order
    with torch.no_grad():
        # Process in smaller batches to avoid memory issues
        batch_size = 3000
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            outputs = model(batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
    
    # If a scaler is provided, inverse transform the predictions
    if scaler:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    
    # Create a copy of the dataframe
    df_with_pred = df.copy()
    df_with_pred.insert(4, 'pred', predictions)
    
    # Calculate metrics for each split
    metrics_dict = {}
    for split in ['train', 'val', 'test', 'all']:
        if split == 'all':
            mask = slice(None)
        else:
            mask = df_with_pred[train_label] == split
            
        y_true = df_with_pred[y_name][mask].values
        y_pred = df_with_pred['pred'][mask].values
        metrics = calculate_metrics(y_true, y_pred)
        metrics_dict[split] = metrics
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame()
    for split, metrics in metrics_dict.items():
        split_metrics = pd.DataFrame([metrics], index=[split])
        metrics_df = pd.concat([metrics_df, split_metrics])
    
    # Reset index to get numeric indices starting from 0
    metrics_df = metrics_df.reset_index()
    metrics_df.index = range(len(metrics_df))
    metrics_df = metrics_df.rename(columns={'index': 'label'})
    
    # Reset index for predictions DataFrame to start from 0
    df_with_pred = df_with_pred.reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    output_dir = '/zhome/32/3/215274/jinlia/project/Adem/data/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both metrics and predictions to a single Excel file with multiple sheets
    output_path = os.path.join(output_dir, f'{dataset_name}_results.xlsx')
    if additional_data is not None:
        output_path = os.path.join(output_dir, f'{dataset_name}_results_additional_{additional_data}.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Save metrics to the 'Metrics' sheet with index
        metrics_df.to_excel(writer, sheet_name='Metrics', index=True)
        
        # Save predictions to the 'Predictions' sheet with index
        df_with_pred.to_excel(writer, sheet_name='Predictions', index=True)
    
    print(f"Saved results to {output_path}")

def create_data_loaders_minimal(dataset_name='hcomb', y_name='Const_Value', 
                              train_label='label', smiles='SMILES', 
                              other_columns=['No'], batch_size=32, scale_y=True, additional_data=25):
    """Create DataLoader objects for train, validation, and test sets with minimal training data
    
    Args:
        dataset_name (str): Name of the dataset
        y_name (str): Name of the target column
        train_label (str): Name of the column containing train/val/test labels
        smiles (str): Name of the SMILES column
        other_columns (list): List of other columns to remove
        batch_size (int): Batch size for DataLoader
        scale_y (bool): Whether to scale the target values
        additional_data (float): Fraction of non-required training data to include
        
    Returns:
        dict: Dictionary containing DataLoaders and related information
    """
    # Load dataset
    df = load_dataset(dataset_name)
    
    # Remove specified columns
    columns_to_remove = [smiles] + other_columns
    feature_df = df.drop(columns=columns_to_remove + [y_name, train_label, 'required'])
    
    # Get features and target
    X = feature_df.values
    print("Initial X shape:", X.shape)
    # Remove columns that are all zeros
    X = X[:, (X != 0).any(axis=0)]
    print("X shape after removing zero columns:", X.shape)
    
    y = df[y_name].values
    labels = df[train_label].values
    required = df['required'].values
    
    # Get required training data
    train_mask = (labels == 'train') & (required == True)
    X_train_required = X[train_mask]
    y_train_required = y[train_mask]
    print(f"Number of required training samples: {len(X_train_required)}")
    
    # Get non-required training data
    train_nonrequired_mask = (labels == 'train') & (required == False)
    X_train_nonrequired = X[train_nonrequired_mask]
    y_train_nonrequired = y[train_nonrequired_mask]
    print(f"Number of non-required training samples: {len(X_train_nonrequired)}")
    
    # Calculate and select additional samples
    n_additional = int(len(X_train_nonrequired) * additional_data/100)
    indices = np.random.permutation(len(X_train_nonrequired))[:n_additional]
    X_train_additional = X_train_nonrequired[indices]
    y_train_additional = y_train_nonrequired[indices]
    print(f"Number of additional samples selected ({additional_data}%): {len(X_train_additional)}")
    
    # Combine required and additional training data
    X_train = np.concatenate([X_train_required, X_train_additional])
    y_train = np.concatenate([y_train_required, y_train_additional])
    print(f"Total training samples: {len(X_train)} (Required: {len(X_train_required)} + Additional: {len(X_train_additional)})")
    
    # Get validation and test data
    X_val = X[labels == 'val']
    X_test = X[labels == 'test']
    y_val = y[labels == 'val']
    y_test = y[labels == 'test']
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\nTarget statistics before scaling:")
    print(f"Training set - Mean: {np.mean(y_train):.4f}, Std: {np.std(y_train):.4f}")
    print(f"Validation set - Mean: {np.mean(y_val):.4f}, Std: {np.std(y_val):.4f}")
    print(f"Test set - Mean: {np.mean(y_test):.4f}, Std: {np.std(y_test):.4f}")
    
    # Create unscaled datasets
    all_dataset = CustomDataset(np.concatenate([X_train, X_val, X_test]), np.concatenate([y_train, y_val, y_test]))  # Using all data for all_loader
    train_dataset_unscaled = CustomDataset(X_train, y_train)
    val_dataset_unscaled = CustomDataset(X_val, y_val)
    test_dataset_unscaled = CustomDataset(X_test, y_test)
    
    print(f"Total samples in all_loader (minimal training set): {len(all_dataset)}")
    
    # Create unscaled loaders
    all_loader = DataLoader(all_dataset, batch_size=batch_size)  # No shuffle for all_loader
    train_loader_unscaled = DataLoader(train_dataset_unscaled, batch_size=batch_size)
    val_loader_unscaled = DataLoader(val_dataset_unscaled, batch_size=batch_size)
    test_loader_unscaled = DataLoader(test_dataset_unscaled, batch_size=batch_size)
    
    # Scale y values if requested
    scaler = None
    if scale_y:
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        print("\nTarget statistics after scaling:")
        print(f"Training set - Mean: {np.mean(y_train_scaled):.4f}, Std: {np.std(y_train_scaled):.4f}")
        print(f"Validation set - Mean: {np.mean(y_val_scaled):.4f}, Std: {np.std(y_val_scaled):.4f}")
        print(f"Test set - Mean: {np.mean(y_test_scaled):.4f}, Std: {np.std(y_test_scaled):.4f}")
        
        # Create scaled datasets
        train_dataset_scaled = CustomDataset(X_train, y_train_scaled)
        val_dataset_scaled = CustomDataset(X_val, y_val_scaled)
        test_dataset_scaled = CustomDataset(X_test, y_test_scaled)
        
        # Create scaled loaders
        train_loader_scaled = DataLoader(train_dataset_scaled, batch_size=batch_size)
        val_loader_scaled = DataLoader(val_dataset_scaled, batch_size=batch_size)
        test_loader_scaled = DataLoader(test_dataset_scaled, batch_size=batch_size)
    else:
        train_loader_scaled = train_loader_unscaled
        val_loader_scaled = val_loader_unscaled
        test_loader_scaled = test_loader_unscaled
    
    # Get input size from the data
    input_size = X_train.shape[1]
    
    # Create original_df with only required training + selected additional + val + test data
    train_required_mask = (labels == 'train') & (required == True)
    train_selected_mask = (labels == 'train') & (required == False) & np.isin(np.arange(len(df)), train_nonrequired_mask.nonzero()[0][indices])
    val_mask = labels == 'val'
    test_mask = labels == 'test'
    
    combined_mask = train_required_mask | train_selected_mask | val_mask | test_mask
    original_df = df[combined_mask].copy()
    
    return {
        'unscaled_loaders': {
            'train': train_loader_unscaled,
            'val': val_loader_unscaled,
            'test': test_loader_unscaled,
            'all': all_loader
        },
        'scaled_loaders': {
            'train': train_loader_scaled,
            'val': val_loader_scaled,
            'test': test_loader_scaled
        },
        'input_size': input_size,
        'original_df': original_df,
        'scaler': scaler
    }

def create_data_loaders(dataset_name='hcomb', y_name='Const_Value', 
                       train_label='label', smiles='SMILES', 
                       other_columns=['No'], batch_size=32, scale_y=True):
    """Create DataLoader objects for train, validation, and test sets"""
    # Load dataset
    df = load_dataset(dataset_name)
    
    # Store original dataframe for later use
    original_df = df.copy()
    
    # Remove specified columns
    columns_to_remove = [smiles] + other_columns
    feature_df = df.drop(columns=columns_to_remove + [y_name, train_label])
    
    # Get features and target
    X = feature_df.values
    print("Initial X shape:", X.shape)
    # remove columns that are all zeros
    X = X[:, (X != 0).any(axis=0)]
    print("X shape after removing zero columns:", X.shape)
    
    y = df[y_name].values
    labels = df[train_label].values
    
    # Split based on predefined labels
    X_train = X[labels == 'train']
    X_val = X[labels == 'val']
    X_test = X[labels == 'test']
    
    y_train = y[labels == 'train']
    y_val = y[labels == 'val']
    y_test = y[labels == 'test']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total samples in all_loader: {len(X)}")
    
    print("\nTarget statistics before scaling:")
    print(f"Training set - Mean: {np.mean(y_train):.4f}, Std: {np.std(y_train):.4f}")
    print(f"Validation set - Mean: {np.mean(y_val):.4f}, Std: {np.std(y_val):.4f}")
    print(f"Test set - Mean: {np.mean(y_test):.4f}, Std: {np.std(y_test):.4f}")
    
    # Create dataset with all data in original order
    all_dataset = CustomDataset(X, y)
    all_loader = DataLoader(all_dataset, batch_size=batch_size)
    
    # Create unscaled datasets and loaders
    train_dataset_unscaled = CustomDataset(X_train, y_train)
    val_dataset_unscaled = CustomDataset(X_val, y_val)
    test_dataset_unscaled = CustomDataset(X_test, y_test)
    
    train_loader_unscaled = DataLoader(train_dataset_unscaled, batch_size=batch_size)
    val_loader_unscaled = DataLoader(val_dataset_unscaled, batch_size=batch_size)
    test_loader_unscaled = DataLoader(test_dataset_unscaled, batch_size=batch_size)
    
    # Scale y values if requested
    scaler = None
    if scale_y:
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    
        y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        print("\nTarget statistics after scaling:")
        print(f"Training set - Mean: {np.mean(y_train_scaled):.4f}, Std: {np.std(y_train_scaled):.4f}")
        print(f"Validation set - Mean: {np.mean(y_val_scaled):.4f}, Std: {np.std(y_val_scaled):.4f}")
        print(f"Test set - Mean: {np.mean(y_test_scaled):.4f}, Std: {np.std(y_test_scaled):.4f}")
        
        # Create scaled datasets and loaders
        train_dataset_scaled = CustomDataset(X_train, y_train_scaled)
        val_dataset_scaled = CustomDataset(X_val, y_val_scaled)
        test_dataset_scaled = CustomDataset(X_test, y_test_scaled)
        
        train_loader_scaled = DataLoader(train_dataset_scaled, batch_size=batch_size)
        val_loader_scaled = DataLoader(val_dataset_scaled, batch_size=batch_size)
        test_loader_scaled = DataLoader(test_dataset_scaled, batch_size=batch_size)
    else:
        train_loader_scaled = train_loader_unscaled
        val_loader_scaled = val_loader_unscaled
        test_loader_scaled = test_loader_unscaled
    
    # Get input size from the data
    input_size = X_train.shape[1]
    
    return {
        'unscaled_loaders': {
            'train': train_loader_unscaled,
            'val': val_loader_unscaled,
            'test': test_loader_unscaled,
            'all': all_loader
        },
        'scaled_loaders': {
            'train': train_loader_scaled,
            'val': val_loader_scaled,
            'test': test_loader_scaled
        },
        'input_size': input_size,
        'original_df': original_df,
        'scaler': scaler
    }

def load_dataset(dataset_name):
    """
    Load dataset from Excel file
    
    Args:
        dataset_name (str): Name of the dataset file without extension
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    path = f"/zhome/32/3/215274/jinlia/project/Adem/data/{dataset_name}.xlsx"
    try:
        df = pd.read_excel(path)
        print(f"Loaded dataset from {path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {path}")
    