##########################################################################################################
#                                                                                                        #
#    Collection of helper function and classes for performing ML-based hyperparameter optimization       #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

# import libraries
import optuna
import argparse
from typing import Dict, Any
import yaml
import importlib
import os
import joblib
from datetime import datetime
from src.model import create_pipeline
import json
import numpy as np
import  random


def hypopt_parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')

    parser.add_argument(
        '--property',
        type=str,
        required=True,
        help='Tag for the property'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model type to optimize (must be defined in config file)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        required=False,
        default='rmse',
        help='Scoring metric to use (must be defined in config file)'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=500,
        help='Number of optimization trials (uses config default if not specified)'
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='tpe',
        help='Sampler to use (uses config default if not specified)'
    )

    parser.add_argument(
        '--path_2_data',
        type=str,
        required=True,
        help='Path to the data file'
    )

    parser.add_argument(
        '--path_2_result',
        type=str,
        required=True,
        help='Path to save the results (metrics and predictions)'
    )

    parser.add_argument(
        '--path_2_model',
        type=str,
        required=True,
        help='Path to save the model and eventual check points'
    )

    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Name of the study for persistence'
    )

    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Database URL for study storage (e.g., sqlite:///optuna.db)'
    )

    parser.add_argument(
        '--no_load_if_exists',
        action='store_false',
        dest='load_if_exists',
        help='Do not load existing study if it exists'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_class_from_path(class_path: str):
    """Dynamically import class from string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_sampler(sampler_config: Dict[str, Any], seed: int = None):
    """Create sampler instance from configuration."""
    sampler_class = get_class_from_path(sampler_config['class'])

    # Get params and update seed if provided
    params = sampler_config.get('params', {}).copy()  # Make a copy to avoid modifying original
    if seed is not None:
        params['seed'] = seed

    return sampler_class(**params)

def create_param_suggest_fn(param_config: Dict[str, Any]):
    """Create appropriate suggest function based on parameter type."""
    if param_config['type'] == 'categorical':
        return lambda trial, name: trial.suggest_categorical(name, param_config['values'])
    elif param_config['type'] == 'float':
        return lambda trial, name: trial.suggest_float(
            name,
            float(param_config['low']),  # Convert to float
            float(param_config['high']), # Convert to float
            log=bool(param_config.get('log', False))  # Convert to boolean
        )
    elif param_config['type'] == 'int':
        return lambda trial, name: trial.suggest_int(
            name,
            int(param_config['low']),    # Convert to int
            int(param_config['high']),   # Convert to int
            log=bool(param_config.get('log', False))  # Convert to boolean
        )
    else:
        raise ValueError(f"Unknown parameter type: {param_config['type']}")

def create_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        property_name: str,
        X_train, y_train,
        X_val, y_val,
        metric_name: str = None,
        sampler_name: str = None,
        n_trials: int = None,
        study_name: str = None,
        storage: str = None,
        load_if_exists: bool = True,
        n_jobs: int = -1,
        seed: int = None,
):
    """
    Create and run hyperparameter optimization using configuration file

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model in config to optimize
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_name: Name of scoring function from config (if None, uses default)
        sampler_name: Name of sampler from config (if None, uses default)
        n_trials: Number of optimization trials (if None, uses default)
    """
    # set no jobs to max

    # Load configuration
    config = load_config(config_path)

    # Get defaults
    defaults = config['default_settings']

    # Setup scoring function
    metric_name = metric_name or defaults['scoring']
    metric_config = config['metric'][metric_name]
    metric_func = get_class_from_path(metric_config['function'])
    direction = metric_config.get('direction', defaults['direction'])

    # Setup sampler
    sampler_name = sampler_name or defaults['sampler']
    sampler_config = config['sampler'][sampler_name]
    sampler = create_sampler(sampler_config, seed=seed)

    # Get number of trials
    n_trials = n_trials or defaults['n_trials']

    # Get model configuration
    model_config = config['models'][model_name]
    model_class = get_class_from_path(model_config['class'])

    # set study name
    study_name = study_name or property_name + '_' + model_name

    # Create parameter suggestion functions
    param_suggest_fns = {
        param_name: create_param_suggest_fn(param_config)
        for param_name, param_config in model_config['param_ranges'].items()
    }

    def objective(trial):
        # Create parameters dictionary using suggestion functions
        params = {
            name: suggest_fn(trial, name)
            for name, suggest_fn in param_suggest_fns.items()
        }

        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train.ravel())

        return metric_func(y_val, model.predict(X_val))

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Setup and run optimization
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name, storage=storage,
                                load_if_exists=load_if_exists)



    # Calculate remaining trials
    existing_trials = len(study.trials)
    if n_trials is not None:
        remaining_trials = n_trials - existing_trials
        n_trials = max(0, remaining_trials)

    print(f"Continuing from existing study with {existing_trials} trials")
    if n_trials > 0:
        print(f"Will run {n_trials} trials")
    else:
        print(f"the desired number of iterations have already been done, consider increasing n_trials")

    # run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Create and fit best model
    best_model = model_class(**study.best_params)
    best_model.fit(X_train, y_train.ravel())


    return study, best_model


def get_parameter_ranges(study):
    """Extract parameter ranges from completed trials"""
    param_ranges = {}

    # Get all parameter names
    if len(study.trials) > 0:
        param_names = study.trials[0].params.keys()

        for param_name in param_names:
            # Get all values tried for this parameter
            values = [t.params[param_name] for t in study.trials if t.params.get(param_name) is not None]

            if len(values) > 0:
                if isinstance(values[0], (int, float)):
                    param_ranges[param_name] = {
                        'min': min(values),
                        'max': max(values),
                        'type': 'float' if isinstance(values[0], float) else 'int'
                    }
                else:
                    # For categorical parameters
                    param_ranges[param_name] = {
                        'values': sorted(list(set(values))),
                        'type': 'categorical'
                    }

    return param_ranges

def save_results(study, fitted_model, fitted_scaler, config_path, model_name, seed,
                 metric_name, model_dir='models/', result_dir='results/'):
    """Save the optimization results, model, and metadata"""
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime('%d%m%Y_%H%M')

    # Base filename
    base_filename = f"{model_name}_{metric_name}_{study.best_value:.3g}_{timestamp}"
    model_path = os.path.join(model_dir, f"{base_filename}_pipeline.joblib")
    results_path = os.path.join(result_dir, f"{base_filename}_results.json")
    trials_path = os.path.join(result_dir, f"{model_name}_{metric_name}_{timestamp}" + "_trials.csv")

    # Create and save pipeline
    pipeline = create_pipeline(fitted_model, fitted_scaler)
    # Create trials DataFrame
    trials_df = study.trials_dataframe()
    # Get parameter ranges
    param_ranges = get_parameter_ranges(study)

    # Save optimization results
    results = {
        'timestamp': timestamp,
        'model_name': model_name,
        'metric_name': metric_name,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'parameter_ranges': param_ranges,
        'config_file': config_path,
        'model_path': model_path,
        'results_path': results_path,
        'trials_path': trials_path,
        'seed': seed,
    }

    # Save results to JSON
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    # save the pipeline
    joblib.dump(pipeline, model_path)
    # Create trials DataFrame
    trials_df.to_csv(trials_path, index=False)

    return results

