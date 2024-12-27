#!/bin/bash

# Run bootstrap analysis for Vc
python config_train_bootstrap_fromRef.py --dataset_name Vc_processed --y_name Vc --save --config_json Vc_params.json

# Run bootstrap analysis for Pc
python config_train_bootstrap_fromRef.py --dataset_name Pc_processed --y_name Pc --save --config_json Pc_params.json

# Run bootstrap analysis for Tc
python config_train_bootstrap_fromRef.py --dataset_name Tc_processed --y_name Tc --save --config_json Tc_params.json 

# Run bootstrap analysis for Omega
python config_train_bootstrap_fromRef.py --dataset_name Omega_processed --y_name Omega --save --config_json Omega_params.json 