import optuna
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from typing import List, Dict
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.gnns import FlexibleMLPAttentiveFP
import optunahub
import argparse
from lightning.pytorch import seed_everything
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from src.features import mol2graph, n_atom_features, n_bond_features
from torch_geometric.loader import DataLoader
import torch
from src.gnns import FlexibleMLPAttentiveFP, AttentiveFP
import torch.nn.functional as F
from src.training import EarlyStopping


class OptunaOptimization:
    def __init__(
            self,
            train_dataset,
            val_dataset,
            n_atom_features_func,  # Function that returns number of atom features
            n_bond_features_func,  # Function that returns number of bond features
            device: str = "cuda",
            n_trials: int = 100,
            study_name: str = "flexiblemlp_optimization"
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Call the functions to get the actual feature dimensions
        self.n_atom_features = n_atom_features_func()
        self.n_bond_features = n_bond_features_func()
        self.device = device
        self.n_trials = n_trials
        self.study_name = study_name

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create a file handler
        fh = logging.FileHandler(f'{study_name}.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    def create_model(self, trial: optuna.Trial) -> FlexibleMLPAttentiveFP:
        """Create model with trial parameters."""
        # Core model parameters
        hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        num_timesteps = trial.suggest_int('num_timesteps', 1, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        # MLP structure
        n_mlp_layers = trial.suggest_int('n_mlp_layers', 1, 4)
        mlp_hidden_dims = []
        for i in range(n_mlp_layers):
            dim = trial.suggest_int(f'mlp_layer_{i}', 16, hidden_channels)
            mlp_hidden_dims.append(dim)

        model = FlexibleMLPAttentiveFP(
            in_channels=self.n_atom_features,
            hidden_channels=hidden_channels,
            out_channels=1,  # Assuming regression task
            edge_dim=self.n_bond_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

        return model.to(self.device)

    def create_dataloader(self, dataset, batch_size: int) -> DataLoader:
        """Create dataloader with given batch size."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(
            self,
            model: FlexibleMLPAttentiveFP,
            loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device: str
    ) -> float:
        """Train for one epoch and return average loss."""
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            loss = F.mse_loss(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)

    def validate(
            self,
            model: FlexibleMLPAttentiveFP,
            loader: DataLoader,
            device: str
    ) -> float:
        """Validate model and return average loss."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch
                )
                loss = F.mse_loss(out.squeeze(), batch.y)
                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Get hyperparameters
        batch_size = trial.suggest_int('batch_size', 16, 128)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        # Create model and optimizer
        model = self.create_model(trial)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Create dataloaders
        train_loader = self.create_dataloader(self.train_dataset, batch_size)
        val_loader = self.create_dataloader(self.val_dataset, batch_size)

        best_val_loss = float('inf')
        patience = trial.suggest_int('patience', 5, 15)
        patience_counter = 0

        for epoch in range(100):  # Maximum 100 epochs per trial
            train_loss = self.train_epoch(model, train_loader, optimizer, self.device)
            val_loss = self.validate(model, val_loader, self.device)

            scheduler.step(val_loss)

            # Log progress
            self.logger.info(f'Trial {trial.number} Epoch {epoch}: '
                             f'Train Loss = {train_loss:.6f}, '
                             f'Val Loss = {val_loss:.6f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f'Trial {trial.number} stopped early at epoch {epoch}')
                break

            # Handle pruning based on intermediate value
            trial.report(val_loss, epoch)
            if trial.should_prune():
                self.logger.info(f'Trial {trial.number} pruned at epoch {epoch}')
                raise optuna.TrialPruned()

        return best_val_loss

    def optimize(self) -> Dict:
        """Run optimization and return best parameters."""
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(self.objective, n_trials=self.n_trials)

        # Log best trial information
        self.logger.info(f"\nBest trial:")
        self.logger.info(f"  Value: {study.best_trial.value:.6f}")
        self.logger.info(f"  Params: {study.best_trial.params}")

        return study.best_trial.params

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='Tc', help='tag of the property of interest')
parser.add_argument('--model', type=str, default='afp', help='name of the GNN model')
parser.add_argument('--path_2_data', type=str, default='data/', help='path to the data')
parser.add_argument('--path_2_result', type=str, default='results/', help='path for storing the results')
parser.add_argument('--path_2_model', type=str, default='models/', help='path for storing the model')
parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility')

args = parser.parse_args()

property_tag = args.property
model_name = args.model
path_2_data = args.path_2_data
path_2_result = args.path_2_result
path_2_model = args.path_2_model
seed = args.seed


path_2_data = path_2_data+'processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
path_2_result = path_2_result+ property_tag+'/gnn/'+model_name+'/'+property_tag+'_result.xlsx'
path_2_model = path_2_model+property_tag+'/gnn/'+model_name+'/'+property_tag

seed_everything(seed)
##########################################################################################################
#%% Data Loading & Preprocessing
##########################################################################################################
# read the data
df = pd.read_excel(path_2_data)
# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']
# construct a scaler
y_scaler = StandardScaler()
y_scaler.fit(df_train[args.property].to_numpy().reshape(-1,1))
# construct a column with the mol objects
df_train = df_train.assign(mol=[Chem.MolFromSmiles(i) for i in df_train['SMILES']])
df_val = df_val.assign(mol=[Chem.MolFromSmiles(i) for i in df_val['SMILES']])
df_test = df_test.assign(mol=[Chem.MolFromSmiles(i) for i in df_test['SMILES']])
# construct molecular graphs
train_dataset = mol2graph(df_train, 'mol', property_tag, y_scaler=y_scaler)
val_dataset = mol2graph(df_val, 'mol', property_tag, y_scaler=y_scaler)
test_dataset = mol2graph(df_test, 'mol', property_tag, y_scaler=y_scaler)


optimizer = OptunaOptimization(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_atom_features_func=n_atom_features,  # Pass the function itself
    n_bond_features_func=n_bond_features,  # Pass the function itself
    device="cuda",
    n_trials=100,
    study_name="flexiblemlp_optimization"
)
best_params = optimizer.optimize()

#%%
from src.training import EarlyStopping
from src.evaluation import predict_property
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlexibleMLPAttentiveFP(
    in_channels=n_atom_features(),
    hidden_channels=best_params['hidden_channels'],
    out_channels=1,
    edge_dim=n_bond_features(),
    num_layers=best_params['num_layers'],
    num_timesteps=best_params['num_timesteps'],
    mlp_hidden_dims=[best_params['mlp_layer_0']],  # Explicitly specify MLP dimensions
    dropout=best_params['dropout'],
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

# Initialize early stopping
early_stopping = EarlyStopping(patience=best_params['patience'], verbose=True)

num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # model training
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
    # model validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            true = batch.y.view(-1, 1)
            val_loss += F.mse_loss(pred, true).item()

    # Print progress
    print(f'Epoch {epoch+1:03d}, Train Loss: {total_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}')

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_loss, model, path_2_model + '_best.pt')
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load best model
model.load_state_dict(torch.load(path_2_model + '_best.pt', weights_only=True))
model = model.to(device)



train_pred, train_true, train_metrics = predict_property(model, train_loader, device, y_scaler)
val_pred, val_true, val_metrics = predict_property(model, val_loader, device, y_scaler)
test_pred, test_true, test_metrics = predict_property(model, test_loader, device, y_scaler)


# Print metrics
print("\nTraining Set Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nValidation Set Metrics:")
for metric, value in val_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

print("\nTest Set Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")