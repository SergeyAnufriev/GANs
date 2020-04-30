#!/bin/bash -l 
. $MODULESHOME/init/bash

#$ -S /bin/bash
#$ -l gpu=1
#$ -l h_rt=0:30:0
#$ -l mem=1G
#$ -l tmpfs=15G
#$ -N GPUJob
#$ -wd /home/zcemg08/Scratch/GANs/gpu_runs



module unload compilers mpi
module load compilers/gnu/4.9.2
module load python/3.7.4
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.5.0.56/cuda-10.0
module load pytorch/1.2.0/gpu

source /home/zcemg08/Scratch/GANs/envs/env4/bin/activate 

wandb login aa5cce262080e13cecdbb604a81606ee881a7af9

python /home/zcemg08/Scratch/GANs/gpu_runs/name.py

line=$(head -n 1 /home/zcemg08/Scratch/GANs/gpu_runs/id.txt)

wandb agent zcemg08/gpu_try2/$line

