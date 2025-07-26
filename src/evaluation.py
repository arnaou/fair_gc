########################################################################################################################
#                                                                                                                      #
#    Collection of helper function and classes for model evaluations                                                   #
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
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import rankdata
import torch

##########################################################################################################
# Define function and classes
##########################################################################################################


def calc_MARE(ym, yp):
    """
    function for calculating the mean absolute relative error
    :param ym: array of target values
    :param yp: array of predicted values
    :return:
    """
    RAE=[]
    pstd=np.std(ym)
    for i in range(0,len(ym)):
        if 0.1 >= ym[i] >= -0.1:
            RAE.append(abs(ym[i]-yp[i])/pstd*100)
        else:
            RAE.append(abs(ym[i]-yp[i])/abs(ym[i]+0.000001)*100)
    mare=np.mean(RAE)
    return mare


def calculate_metrics(y_target, y_pred):
    """
    function for calculating a wide range of performance metrics
    :param y_target: array of target values
    :param y_pred: array of predicted values
    :return: dict of performance metrics (r2, rmse, mse, mare, mae)
    """

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
    # collect the output
    out = {'r2': r2, 'rmse': rmse ,'mse': mse, 'mare': mare, 'mae': mae}
    return out


def identify_outliers_ecdf(values, alpha):
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
    # calculate the thresholds
    lower_threshold = alpha
    upper_threshold = 1-alpha

    # Calculate ECDF values
    n = len(values)
    ecdf = rankdata(values) / n

    # Create outlier mask
    outlier_mask = (ecdf < lower_threshold) | (ecdf > upper_threshold)
    return outlier_mask



def evaluate_gnn(model, loader, device, y_scaler=None, tag='afp'):
    """
    function for using a GNN for prediction
    :param model: rained PyG GNN model
    :param loader: Dataloader
    :param device: cpu or cuda
    :param y_scaler: the scaler for the target value
    :param tag: dictates how the model takes inputs
    :return:
    """
    # set the model in evaluation mode
    model.eval()
    # initialize the prediction and true values
    predictions = []
    true_values = []

    # unpack loaders and perform predictions
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if tag == 'afp':
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            elif tag in ['mpnn','megnet', 'groupgat']:
                pred = model(batch)
            true = batch.y.view(-1, 1)

            # Store predictions and true values
            predictions.extend(pred.cpu().numpy())
            true_values.extend(true.cpu().numpy())

    # convert to numpy
    predictions = np.array(predictions).reshape(-1, 1)
    true_values = np.array(true_values).reshape(-1, 1)

    # Inverse transform if scaler was used
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
        true_values = y_scaler.inverse_transform(true_values)

    # Calculate metrics
    metrics = calculate_metrics(true_values, predictions)

    return predictions, true_values, metrics


def evaluate_mlp(model, loader, device, y_scaler=None):
    """
    function for using a GNN for prediction
    :param model: rained PyG GNN model
    :param loader: Dataloader
    :param device: cpu or cuda
    :param y_scaler: the scaler for the target value
    :return:
    """
    # set the model in evaluation mode
    model.eval()
    # initialize the prediction and true values
    predictions = []
    true_values = []

    # unpack loaders and perform predictions
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(x)
            true = y.view(-1, 1)

            # Store predictions and true values
            predictions.extend(pred.cpu().numpy())
            true_values.extend(true.cpu().numpy())

    # convert to numpy
    predictions = np.array(predictions).reshape(-1, 1)
    true_values = np.array(true_values).reshape(-1, 1)

    # Inverse transform if scaler was used
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
        true_values = y_scaler.inverse_transform(true_values)

    # Calculate metrics
    metrics = calculate_metrics(true_values, predictions)

    return predictions, true_values, metrics