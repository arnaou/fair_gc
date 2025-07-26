# LSBATCH: User input
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q ktprosys
### -- set the job Name --
#BSUB -J GC-ML/MLP
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- set walltime limit: hh:mm --
#BSUB -W 240:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address, 
# if you want to receive e-mail notifications on a non-default address 
#BSUB -u arnaou@kt.dtu.dk 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o hpc_outputs/Output_%J.log 
#BSUB -e hpc_outputs/Error_%J.err 
# -- end of LSF options --


# COMMANDS YOU WANT EXECUTED
# load virtual environment
# it includes loading python3
source $HOME/miniconda3/bin/activate critprops


# run application
#python3 optuna_ml_v1.py --property Tc --config_file ml_hyperopt_config.yaml --model svr --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --sampler auto --n_jobs 3
python3 optuna_mlp.py --property Pc --config_file mlp_hyperopt_config.yaml --model mlp --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --sampler auto --n_jobs 3 --split_type butina_min
