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




