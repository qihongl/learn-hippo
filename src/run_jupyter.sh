#!/bin/bash
#SBATCH -t 00:59:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH --mem-per-cpu 8G
#SBATCH --ntasks-per-socket=1
#SBATCH --job-name jl
#SBATCH --output jupyter-log-%J.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qlu@princeton.edu


## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    -----------------------------------------------------------------
    ssh -N -f -L $ipnport:$ipnip:$ipnport $USER@tigergpu.princeton.edu
    localhost:$ipnport
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter server
#jupyter notebook --no-browser --port=$ipnport --ip=$ipnip
jupyter lab --no-browser --port=$ipnport --ip=$ipnip
