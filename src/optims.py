

def print_callback(study, trial):
    """
    call back function for optuna
    :param study:
    :param trial:
    :return:
    """
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")
