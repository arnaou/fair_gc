#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J hyp_rhoc
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,  
# if you want to receive e-mail notifications on a non-default address
##BSUB -u arnaou@kt.dtu.dk
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_outputs/Output_%J.log 
#BSUB -e hpc_outputs/Error_%J.err 
# -- end of LSF options --

### nvidia-smi
source $HOME/miniconda3/bin/activate critprops

#python3 HyPE/optimization_run_cluster.py --config_file config_tc.yaml --samples 5
#python3 megnet_hyperopt.py --property Omega --config_file megnet_hyperopt_config.yaml --model megnet --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3
#python3 groupgat_hyperopt.py --property Tc --config_file groupgat_hyperopt_config.yaml --model groupgat --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3
python3 groupgat_hyperopt.py --property Pc --config_file groupgat_hyperopt_config.yaml --model groupgat --n_trials 2500 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3
