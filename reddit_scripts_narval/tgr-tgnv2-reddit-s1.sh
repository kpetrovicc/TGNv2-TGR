#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=4-10:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=1           # CPU cores/threads
#SBATCH --gres=gpu:a100:1                # number of GPU(s) per node
#SBATCH --mem=100G                   # memory (per node)
#SBATCH --mail-user=petrovickatarina684@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=LPP_tgr_tgnv2_gat_reddit
#SBATCH --output=outlog/%x-%j.log

cd ~/projects/def-rrabba/kpetr
module purge 
module load StdEnv/2020
module load python/3.10
source tgbkp/bin/activate 
module purge 
module load StdEnv/2020
module load python/3.10
source tgbkp/bin/activate 

seed=1
lr=1e-4
epochs=50
data="tgbn-reddit"
dataset="tgbn-reddit"

echo " >>> DATA: $data"
echo " >>> Seed: $seed"

echo "===================================================================================="
echo "===================================================================================="
echo ""
echo " ***** TGN: $data *****"
echo ""
python TGNv2-TGR/train-tgbn-nodeproppred-tgr-reddit.py --seed "$seed" --epochs "$epochs" --lr="$lr" --dataset "$dataset"
echo "===================================================================================="
echo "===================================================================================="

