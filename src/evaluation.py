import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error




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


