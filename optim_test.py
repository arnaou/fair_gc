# optimizer.py
import argparse
import yaml
import optuna
from model_utils import (
    load_config,
    get_class_from_path,
    create_param_suggest_fn,
    create_sampler
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')

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
        required=True,
        help='Scoring metric to use (must be defined in config file)'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=None,
        help='Number of optimization trials (uses config default if not specified)'
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default=None,
        help='Sampler to use (uses config default if not specified)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the data file'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )

    return parser.parse_args()


def create_hyperparameter_optimizer(
        config_path: str,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
        scoring_name: str,
        sampler_name: str = None,
        n_trials: int = None
):
    # Load configuration
    config = load_config(config_path)

    # Get defaults
    defaults = config['default_settings']

    # Validate model exists in config
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in config file")

    # Validate metric exists in config
    if scoring_name not in config['scoring']:
        raise ValueError(f"Metric '{scoring_name}' not found in config file")

    # Setup scoring function
    scoring_config = config['scoring'][scoring_name]
    scoring_func = get_class_from_path(scoring_config['function'])
    direction = scoring_config.get('direction', defaults['direction'])

    # Setup sampler
    sampler_name = sampler_name or defaults['sampler']
    if sampler_name not in config['samplers']:
        raise ValueError(f"Sampler '{sampler_name}' not found in config file")
    sampler_config = config['samplers'][sampler_name]
    sampler = create_sampler(sampler_config)

    # Get number of trials
    n_trials = n_trials or defaults['n_trials']

    # Get model configuration
    model_config = config['models'][model_name]
    model_class = get_class_from_path(model_config['class'])

    # Create parameter suggestion functions
    param_suggest_fns = {
        param_name: create_param_suggest_fn(param_config)
        for param_name, param_config in model_config['param_ranges'].items()
    }

    def objective(trial):
        params = {
            name: suggest_fn(trial, name)
            for name, suggest_fn in param_suggest_fns.items()
        }

        model = model_class(**params)
        model.fit(X_train, y_train.ravel())

        return scoring_func(y_val, model.predict(X_val))

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study


def main():
    args = parse_arguments()

    # Load and prepare data
    # This is an example - modify according to your data loading needs
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np

    # read the data
    df = pd.read_excel(path_2_data)
    # construct list of columns indices
    columns = [str(i) for i in range(1, 425)]
    # remove zero columns
    df = remove_zero_one_sum_rows(df, columns)
    # construct group ids
    grp_idx = [str(i) for i in range(1, 425)]
    # retrieve indices of available groups
    idx_avail = find_nonzero_columns(df, ['SMILES', property_tag, 'label', 'No'])
    # split the indices into 1st, 2nd and 3rd order groups
    idx_mg1, idx_mg2, idx_mg3 = split_indices(idx_avail)
    # extract the number of available groups in each order
    n_mg1 = len(idx_mg1)
    n_mg2 = len(idx_mg2)
    n_mg3 = len(idx_mg3)
    n_pars = n_mg1 + n_mg2 + n_mg3
    # split the data
    df_train = df[df['label'] == 'train']
    df_val = df[df['label'] == 'val']
    df_test = df[df['label'] == 'test']
    # extract feature vectors and targets for training
    X_train = df_train.loc[:, idx_avail].to_numpy()
    y_train = {'true': df_train[property_tag].to_numpy()}
    # extract feature vectors and targets for validation
    X_val = df_val.loc[:, idx_avail].to_numpy()
    y_val = {'true': df_val[property_tag].to_numpy()}
    # extract feature vectors and targets for testing
    X_test = df_test.loc[:, idx_avail].to_numpy()
    y_test = {'true': df_test[property_tag].to_numpy()}
    # # scaling the target
    scaler = StandardScaler()
    y_train['scaled'] = scaler.fit_transform(y_train['true'])
    y_val['scaled'] = scaler.transform(y_val['true'])
    y_test['scaled'] = scaler.transform(y_test['true'])


    # Run optimization
    study = create_hyperparameter_optimizer(
        config_path=args.config_file,
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scoring_name=args.metric,
        sampler_name=args.sampler,
        n_trials=args.n_trials
    )

    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")

    # Optionally save results
    import json
    from datetime import datetime

    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model': args.model,
        'metric': args.metric,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials or study.n_trials
    }

    output_file = f"results_{args.model}_{args.metric}_{results['timestamp']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
