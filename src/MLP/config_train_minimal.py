import os
import nni
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import MLP
from config_data_utils import create_data_loaders, device, calculate_metrics, save_predictions,create_data_loaders_minimal
import argparse
from torchkeras import summary

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Or suppress all warnings
warnings.filterwarnings("ignore")

# Your code here


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader, scaler=None):
    model.eval()
    predictions = []
    actuals = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            
            # Convert to numpy for scaling
            outputs = outputs.cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            
            # Inverse transform predictions if scaler is provided
            if scaler:
                print("mean of outputs before scaling:", np.mean(outputs))
                outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).ravel()
                print("mean of outputs after scaling:", np.mean(outputs))
                # Don't scale batch_y since we're using unscaled loader
            
            predictions.extend(outputs)
            actuals.extend(batch_y)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return calculate_metrics(actuals, predictions)

def train_model1(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=500, dataset_name='hcomb'):
    best_val_loss = float('inf')
    patience = 60
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # train_loss /= (len(train_loader)/train_loader.batch_size)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        
        # Report intermediate result to NNI
        nni.report_intermediate_result(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'/zhome/32/3/215274/jinlia/project/Adem/model_cache/{dataset_name}_best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.6f}')
            break
    
    return best_val_loss

def parse_init_json(data):
    """Parse the tree-structured hyperparameters into MLP configuration"""
    params = {
        'layer_sizes': [],
        'dropouts': [],
        'activations': [],
        'learning_rate': data['learning_rate']
    }
    
    # Process first three layers (optional layers without dropout)
    for i in range(3):  # for layers 0-2
        layer_key = f'layer{i}'
        layer_config = data[layer_key]
        
        if layer_config['_name'] == 'Empty':
            continue
        elif layer_config['_name'] == 'Dense':
            params['layer_sizes'].append(int(layer_config['size']))
            params['dropouts'].append(0.0)  # No dropout for layers 0-2
            params['activations'].append(layer_config['activation'])
    
    # Add layer3 (mandatory layer) using flat parameters
    params['layer_sizes'].append(int(data['layer3_size']))
    params['dropouts'].append(data['layer3_dropout'])
    params['activations'].append(data['layer3_activation'])
    
    params['num_layers'] = len(params['layer_sizes'])
    return params

def get_default_params(args):
    """Default parameters when not using NNI, matching the search space structure"""
    import json
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), f'config/{args.config_json}')
    
    try:
        with open(config_path, 'r') as f:
            print("using config file:", config_path)
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using hardcoded defaults.")
        return {
            "layer0": {
                "_name": "Dense", 
                "size": 56,
                "activation": "leaky_relu"
            },
            "layer1": {
                "_name": "Dense",
                "size": 56, 
                "activation": "relu"
            },
            "layer2": {
                "_name": "Dense",
                "size": 96,
                "activation": "leaky_relu"
            },
            "layer3_size": 56,
            "layer3_dropout": 0.001689586605154624,
            "layer3_activation": "leaky_relu",
            "learning_rate": 0.0009080167558708478
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP model with specified dataset parameters')
    parser.add_argument('--dataset_name', type=str, default='Vc_processed_minimal', help='Name of the dataset')
    parser.add_argument('--y_name', type=str, default='Vc', help='Target variable column name')
    parser.add_argument('--scale_y', type=bool, default=True, help='Scale the target variable')
    parser.add_argument('--batch_size', type=int, default=3200, help='Batch size')
    parser.add_argument('--train_label', type=str, default='label', help='Column name for train/val/test split')
    parser.add_argument('--smiles', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--other_columns', nargs='+', default=['No','required'], help='Additional columns to remove')
    parser.add_argument('--save', action='store_true', help='Save predictions to file if specified')
    parser.add_argument('--additional_data', type=int, default=0, help='Additional data to add')
    parser.add_argument('--save_best_model',default=True, help='Save the best model')
    parser.add_argument('--config_json', type=str, default='Vc_params.json', help='Path to the config json file')
    return parser.parse_args()


def get_cpu_model():
    with open('/proc/cpuinfo') as f:
        for line in f:
            if "model name" in line:
                return line.split(":")[1].strip()

def main():


    cpu_model = get_cpu_model()
    print(f"CPU Model: {cpu_model}")

    args = parse_args()
    
    # Get parameters from NNI if available, otherwise use defaults
    params = nni.get_next_parameter()
    is_nni = True if params else False
    if is_nni:
        print("Using NNI parameters:", params)
    else:
        params = get_default_params(args)
        print("Using default parameters:", params)
    
    # Parse the parameters into MLP structure
    mlp_params = parse_init_json(params)
    print("Parsed MLP parameters:", mlp_params)
    
    # Create data loaders with the parsed arguments
    data_dict = create_data_loaders_minimal(
        dataset_name=args.dataset_name,
        y_name=args.y_name,
        train_label=args.train_label,
        smiles=args.smiles,
        other_columns=args.other_columns,
        batch_size=args.batch_size,
        scale_y= args.scale_y,
        additional_data=args.additional_data
    )
    
    # Get both scaled and unscaled loaders
    train_loader = data_dict['scaled_loaders']['train']
    val_loader = data_dict['scaled_loaders']['val']
    test_loader = data_dict['scaled_loaders']['test']
    input_size = data_dict['input_size']
    scaler = data_dict['scaler']
    
    # Create model with parsed parameters
    model = MLP(mlp_params, input_size=input_size).to(device)
    # summary in cpu
    summary(model, input_shape=(1, input_size), input_dtype=torch.FloatTensor)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    def rmse_loss(y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse)
    criterion = rmse_loss
    # criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=True
    )
    
    # Train the model using scaled data
    best_val_loss = train_model1(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, epochs=1000, dataset_name=args.dataset_name
    )
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(f'/zhome/32/3/215274/jinlia/project/Adem/model_cache/{args.dataset_name}_best_model.pth'
                                     )) 
    #save the best model if save_best_model is True
    if args.save_best_model:
        torch.save(model.state_dict(), f'/zhome/32/3/215274/jinlia/project/Adem/data/results/{args.dataset_name}{args.additional_data}_best_model.pth')
    
    print("trained with scaled data where scale_y is", args.scale_y)
    # Evaluate on unscaled data
    print("\nEvaluating on unscaled data:")
    for name, loader in [
        ("Train", data_dict['unscaled_loaders']['train']),
        ("Validation", data_dict['unscaled_loaders']['val']),
        ("Test", data_dict['unscaled_loaders']['test']),
        ("All", data_dict['unscaled_loaders']['all'])
    ]:
        # take the rmse on unscaled validation set as the final result  
        

        metrics = evaluate_model(model, loader, scaler)
        print(f"\nMLP {name} on dataset {args.dataset_name} Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        if name == "Validation":
            final_result = metrics['rmse']
            print("report final result:", final_result)
            if is_nni:
                nni.report_final_result(final_result)
           
    
    # Save predictions if requested
    if args.save:
        save_predictions(
            data_dict['original_df'],
            model,
            device,
            dataset_name=args.dataset_name,
            y_name=args.y_name,
            train_label=args.train_label,
            smiles=args.smiles,
            other_columns=args.other_columns,
            scaler=scaler,
            additional_data=args.additional_data
        )

if __name__ == "__main__":
    main() 