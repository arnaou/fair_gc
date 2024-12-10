import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import random
import os
import optuna
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2torchdata, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
from src.gnns import AttentiveFP
import torch.nn.functional as F
from src.training import EarlyStopping, seed_everything
from src.evaluation import predict_property


def define_model_trial(trial):
    """Define the hyperparameter search space."""
    return {
        'hidden_channels': trial.suggest_int('hidden_channels', 32, 300),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'num_timesteps': trial.suggest_int('num_timesteps', 1, 4),
        'num_mlp_layers': trial.suggest_int('num_mlp_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [2048]),
    }


def objective(trial, args, df_train, df_val, y_scaler):
    """Optuna objective function."""
    # Get hyperparameters for this trial
    params = define_model_trial(trial)

    # Set seed
    seed_everything(42)

    # Create datasets and loaders
    train_dataset = mol2torchdata(df_train, 'mol', args.property, y_scaler=y_scaler)
    val_dataset = mol2torchdata(df_val, 'mol', args.property, y_scaler=y_scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Initialize model with trial parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentiveFP(
        in_channels=n_atom_features(),
        hidden_channels=params['hidden_channels'],
        out_channels=1,
        edge_dim=n_bond_features(),
        num_layers=params['num_layers'],
        num_timesteps=params['num_timesteps'],
        num_mlp_layers=params['num_mlp_layers'],
        dropout=params['dropout']
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=8
    )

    # Training loop
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=15, verbose=False)

    for epoch in range(50):  # Maximum 50 epochs per trial
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            true = batch.y.view(-1, 1)
            loss = F.mse_loss(pred, true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                true = batch.y.view(-1, 1)
                val_loss += F.mse_loss(pred, true).item()

        val_loss = val_loss / len(val_loader)

        # Update scheduler and check early stopping
        scheduler.step(val_loss)
        early_stopping(val_loss, model, f'temp_model_{trial.number}.pt')

        if early_stopping.early_stop:
            break

        # Report intermediate value
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Clean up temporary model file
    if os.path.exists(f'temp_model_{trial.number}.pt'):
        os.remove(f'temp_model_{trial.number}.pt')

    return val_loss


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', type=str, default='Omega', help='tag of the property of interest')
    parser.add_argument('--model', type=str, default='afp', help='name of the GNN model')
    parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
    parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
    parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
    parser.add_argument('--n_trials', type=int, default=1000, help='number of optimization trials')
    args = parser.parse_args()

    # Setup paths
    path_2_data = args.path_2_data + 'processed/' + args.property + '/' + args.property + '_processed.xlsx'

    # Load and preprocess data
    df = pd.read_excel(path_2_data).sort_index()
    df_train = df[df['label'] == 'train']
    df_val = df[df['label'] == 'val']

    # Scale target variable
    y_scaler = StandardScaler()
    y_scaler.fit(df_train[args.property].to_numpy().reshape(-1, 1))

    # Create mol objects
    df_train = df_train.assign(mol=[Chem.MolFromSmiles(i) for i in df_train['SMILES']])
    df_val = df_val.assign(mol=[Chem.MolFromSmiles(i) for i in df_val['SMILES']])

    # Create study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42,
                           n_startup_trials=10,    # Adjust number of random trials
                           multivariate=True,      # Toggle multivariate optimization
                           constant_liar=True      # Toggle constant liar algorithm
                           ))

    # Optimize
    study.optimize(
        lambda trial: objective(trial, args, df_train, df_val, y_scaler),
        n_trials=args.n_trials,
        timeout=None
    )

    # Print results
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save study results
    study_path = args.path_2_result + args.property + '/optimization_results.pkl'
    os.makedirs(os.path.dirname(study_path), exist_ok=True)
    optuna.save_study(study, study_path)

    # Create results DataFrame
    trials_df = study.trials_dataframe()
    trials_df.to_csv(args.path_2_result + args.property + '/optimization_trials.csv')


if __name__ == "__main__":
    main()