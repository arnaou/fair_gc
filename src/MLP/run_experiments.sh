   #!/bin/bash

   for i in {0..100..5}
   do
       python config_train_minimal.py --dataset_name Omega_processed_minimal --y_name Omega --config_json Omega_params.json --additional_data $i --save
       python config_train_minimal.py --dataset_name Tc_processed_minimal --y_name Tc --config_json Tc_params.json --additional_data $i --save
       python config_train_minimal.py --dataset_name Vc_processed_minimal --y_name Vc --config_json Vc_params.json --additional_data $i --save    
       python config_train_minimal.py --dataset_name Pc_processed_minimal --y_name Pc --config_json Pc_params.json --additional_data $i --save    
   done
