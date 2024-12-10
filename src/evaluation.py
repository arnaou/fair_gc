import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import rankdata
import torch


#Calculate MARE
def calc_MARE(ym, yp):
    RAE=[]
    pstd=np.std(ym)
    for i in range(0,len(ym)):
        if 0.1 >= ym[i] >= -0.1:
            RAE.append(abs(ym[i]-yp[i])/pstd*100)
        else:
            RAE.append(abs(ym[i]-yp[i])/abs(ym[i]+0.000001)*100)
    mare=np.mean(RAE)
    return mare


def calculate_metrics(y_target, y_pred, n_params=None):

    # length of data
    N = y_target.shape[0]
    # degree of freedom
    #v = N - n_params
    # calculate R2
    r2 = r2_score(y_target, y_pred)
    # calculate rmse
    rmse = root_mean_squared_error(y_target, y_pred)
    # calculate MSE
    mse = mean_squared_error(y_target, y_pred)
    # calculate MARE
    mare = calc_MARE(y_target, y_pred)
    # calculate MAE
    mae = mean_absolute_error(y_target, y_pred)

    out = {'r2': r2, 'rmse': rmse ,'mse': mse, 'mare': mare, 'mae': mae}

    return out


def identify_outliers_ecdf(values, lower_threshold=0.025, upper_threshold=0.975):
    """
    Identify outliers based on ECDF thresholds.

    Parameters:
    -----------
    values : array-like
        Values to check for outliers
    lower_threshold : float
        Lower ECDF threshold (default: 0.025 for 2.5th percentile)
    upper_threshold : float
        Upper ECDF threshold (default: 0.975 for 97.5th percentile)

    Returns:
    --------
    outlier_mask : numpy array
        Boolean mask indicating outlier points
    """
    # Calculate ECDF values
    n = len(values)
    ecdf = rankdata(values) / n

    # Create outlier mask
    outlier_mask = (ecdf < lower_threshold) | (ecdf > upper_threshold)
    return outlier_mask



def predict_property(model, loader, device, y_scaler=None):
    """
    Make predictions using the trained model and calculate metrics

    Args:
        model: Trained AttentiveFP model
        loader: PyTorch Geometric DataLoader
        device: torch device
        y_scaler: StandardScaler used for target scaling

    Returns:
        predictions: Numpy array of predictions
        true_values: Numpy array of true values
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            true = batch.y.view(-1, 1)

            # Store predictions and true values
            predictions.extend(pred.cpu().numpy())
            true_values.extend(true.cpu().numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Inverse transform if scaler was used
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
        true_values = y_scaler.inverse_transform(true_values)

    # Calculate metrics
    metrics = calculate_metrics(true_values, predictions)

    return predictions, true_values, metrics