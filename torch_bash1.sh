#!/bin/bash 
. $MODULESHOME/init/bash

#$ -S /bin/bash
#$ -l gpu=1
#$ -l h_rt=0:10:0
#$ -l mem=1G
#$ -l tmpfs=15G
#$ -N GPUJob
#$ -wd ~/PhD/tutorials/videos/intro

WANDB_API_KEY=$aa5cce262080e13cecdbb604a81606ee881a7af9
export WANDB_API_KEY


module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/3.6
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-10.0
module load pytorch/1.2.0/gpu

source ~/envs/env3/bin/activate 

python ~/PhD/tutorials/videos/intro/gan_torch.py
