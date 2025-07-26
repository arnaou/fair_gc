# LSBATCH: User input
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q ktprosys
### -- set the job Name --
#BSUB -J ICP_classyfire
### -- ask for number of cores (default: 1) -- 
#BSUB -n 7
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
#BSUB -o Output_%J.log 
#BSUB -e Error_%J.err 
# -- end of LSF options --


# COMMANDS YOU WANT EXECUTED
# load virtual environment
# it includes loading python3
source $HOME/miniconda3/bin/activate qspr


# run application
python3 classification_arnaou.py