import torch
import os
import random
import numpy as np
import torch_geometric


def seed_everything(TORCH_SEED):
    # Basic seeds
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)

    # CUDA seeds
    torch.cuda.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)

    # PyG seed
    torch_geometric.seed_everything(TORCH_SEED)

    # Backend control
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # Add this line

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)

    # Set CUDA environment
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Force deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Early stopping to stop training when validation loss doesn't improve.

        Args:
            patience (int): Number of epochs to wait after min has been hit
            min_delta (float): Minimum change in monitored value to qualify as improvement
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss