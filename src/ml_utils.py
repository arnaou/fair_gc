##########################################################################################################
#                                                                                                        #
#    Collection of helper function and classes for performing creating ML-based pipelines                #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin




class PostScaledRegressor(BaseEstimator, RegressorMixin):
    """
    a class for creating a regressor for ML-based models
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        # Get the actual model instance from the class if needed
        if isinstance(self.model, type):
            print("Model is a class, not an instance")
            # Try to get the actual model instance
            if hasattr(self.model, '_instance'):
                model = self.model._instance
            else:
                raise ValueError("Cannot find model instance")
        else:
            model = self.model

        # Make prediction
        if isinstance(X, pd.DataFrame):
            X = X.values

        scaled_pred = self.model.predict(X)
        return self.scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()


def create_kernel(params):
    """Create GPR kernel based on parameters"""
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

    kernel_type = params['kernel_type']
    length_scale = params['length_scale']
    noise_level = params['noise_level']
    constant_value = params['constant_value']

    if kernel_type == 'RBF':
        kernel = ConstantKernel(constant_value) * RBF(length_scale) + WhiteKernel(noise_level)
    else:  # Matern
        nu = params['nu']
        kernel = ConstantKernel(constant_value) * Matern(length_scale, nu=nu) + WhiteKernel(noise_level)

    return kernel

def create_model(model_class, params, seed=None):
   """Create model instance with proper seed handling"""
   if model_class.__name__ == 'GaussianProcessRegressor':
       kernel = create_kernel(params)
       gpr_params = {k: v for k, v in params.items()
                    if k not in ['kernel_type', 'length_scale', 'noise_level',
                               'constant_value', 'nu']}
       return model_class(kernel=kernel, random_state=seed, **gpr_params)
   elif model_class.__name__ in ['RandomForestRegressor', 'DecisionTreeRegressor']:
       return model_class(random_state=seed, **params)
   else:
       return model_class(**params)  # For models like SVR that don't use random_state

def create_pipeline(fitted_model, fitted_scaler):
    """Create pipeline with pre-fitted model and scaler"""
    return PostScaledRegressor(fitted_model, fitted_scaler)

def load_model(model_path):
    """Load the saved pipeline"""
    return joblib.load(model_path)

def predict_new_data(model_path, X_new):
    """Load model and make predictions"""
    pipeline = joblib.load(model_path)
    if pipeline is not None:
        # Get scaled predictions
        unscaled_predictions = pipeline.predict(X_new)

        # If you need unscaled predictions
        scaled_predictions = pipeline.model.predict(X_new)

        return scaled_predictions, unscaled_predictions
    return None, None



