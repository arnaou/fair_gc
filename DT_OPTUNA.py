# import the packages
import pandas as pd


from src.data import remove_zero_one_sum_rows
from src.optims import print_callback
from src.splits import find_nonzero_columns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, root_mean_squared_error
import optuna
from optuna.samplers import TPESampler
from src.optims import print_callback
import joblib


# import the data
property_tag = 'Omega'
path_to_data = 'data/processed/'+property_tag+'/'+property_tag+'_processed.xlsx'
df = pd.read_excel(path_to_data)

# remove the zero elements
columns = [str(i) for i in range(1, 425)]
df= remove_zero_one_sum_rows(df, columns)

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


def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 35),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'criterion': trial.suggest_categorical('criterion',
                                               ['squared_error', 'friedman_mse', 'absolute_error']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random'])
    }

    # Create and train model
    model = DecisionTreeRegressor(random_state=42, **params)
    model.fit(X_train, y_train.ravel())

    # Calculate validation score
    val_pred = model.predict(X_val)
    score = mean_absolute_error(y_val, val_pred)

    return score


# Create study object and optimize
sampler = TPESampler(seed=10, multivariate=True)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=500, show_progress_bar=True)#, callbacks=[print_callback])

# Print optimization results
print("\nBest parameters:", study.best_params)
print("Best validation score:", study.best_value)

# Train final model with best parameters
best_model = DecisionTreeRegressor(random_state=42, **study.best_params)
best_model.fit(X_train, y_train.ravel())

#%% rescale

#%% save study

# Save study results
joblib.dump(study, "decision_tree_optimization.pkl")

# Load study results
loaded_study = joblib.load("decision_tree_optimization.pkl")

# Get trials dataframe
trials_df = study.trials_dataframe()
print("\nTop 5 trials:")
print(trials_df.sort_values('value', ascending=True).head())

#%%
train_score = r2_score(y_train, best_model.predict(X_train))
val_score = r2_score(y_val, best_model.predict(X_val))
test_score = r2_score(y_test, best_model.predict(X_test))

print("\nModel Performance (R2):")
print(f"Training: {train_score:.4f}")
print(f"Validation: {val_score:.4f}")
print(f"Test: {test_score:.4f}")


train_score = mean_absolute_error(y_train, best_model.predict(X_train))
val_score = mean_absolute_error(y_val, best_model.predict(X_val))
test_score = mean_absolute_error(y_test, best_model.predict(X_test))
print("\nModel Performance (MAE):")
print(f"Training: {train_score:.4f}")
print(f"Validation: {val_score:.4f}")
print(f"Test: {test_score:.4f}")