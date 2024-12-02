# import the packages
import pandas as pd


from src.data import remove_zero_one_sum_rows
from src.splits import find_nonzero_columns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, root_mean_squared_error
import optuna
from optuna.samplers import TPESampler
from src.optims import create_hyperparameter_optimizer


# import the data
property_tag = 'Omega'
path_to_data = 'data/processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
df = pd.read_excel(path_to_data)

# remove the zero elements
columns = [str(i) for i in range(1, 425)]
# df= remove_zero_one_sum_rows(df, columns)

# split the data
df_train = df[df['label']=='train']
df_val = df[df['label']=='val']
df_test = df[df['label']=='test']

# extract feature vectors and targets
X_train = df_train.loc[:,'1':].to_numpy()
y_train = df_train[property_tag].to_numpy().reshape(-1,1)

X_val = df_val.loc[:,'1':].to_numpy()
y_val = df_val[property_tag].to_numpy().reshape(-1,1)

X_test = df_test.loc[:,'1':].to_numpy()
y_test = df_test[property_tag].to_numpy().reshape(-1,1)

# scaling the data
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)


param_ranges = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': {'low': 1e-1, 'high': 1e3, 'log': True},
    'gamma': {'low': 1e-4, 'high': 1e0, 'log': True},
    'epsilon': {'low': 0, 'high': 0.3, 'log': False}
}

study = create_hyperparameter_optimizer(
    model_class=SVR,
    param_ranges=param_ranges,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scoring_func=mean_absolute_error,
    n_trials=50
)

# Get best parameters
#best_params = study.best_params
#print("Best parameters:", best_params)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)


#%%
from src.evaluation import calculate_metrics
best_kernel = study.best_params['kernel']
best_C = study.best_params['C']
best_gamma = study.best_params['gamma']
best_epsilon = study.best_params['epsilon']

# Create best model
best_svr = SVR(
    kernel=best_kernel,
    C=best_C,
    gamma=best_gamma,
    epsilon=best_epsilon
)

# Fit and evaluate
best_svr.fit(X_train, y_train.ravel())


#%% rescale


#%%
train_score = r2_score(y_train, best_svr.predict(X_train))
val_score = r2_score(y_val, best_svr.predict(X_val))
test_score = r2_score(y_test, best_svr.predict(X_test))

print("\nModel Performance (R2):")
print(f"Training: {train_score:.4f}")
print(f"Validation: {val_score:.4f}")
print(f"Test: {test_score:.4f}")


train_score = mean_absolute_error(y_train, best_svr.predict(X_train))
val_score = mean_absolute_error(y_val, best_svr.predict(X_val))
test_score = mean_absolute_error(y_test, best_svr.predict(X_test))
print("\nModel Performance (MAE):")
print(f"Training: {train_score:.4f}")
print(f"Validation: {val_score:.4f}")
print(f"Test: {test_score:.4f}")