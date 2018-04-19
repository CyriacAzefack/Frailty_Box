#!/bin/bash
#SBATCH --job-name=KA_Simulation_100_rep
#SBATCH --mail-user=cyriac.azefack@emse.fr
#SBATCH --mail-type=ALL
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
unset SLURM_GTIDS

DATASET_NAME=KC

echo ------------------------------------------------------
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST
echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR
echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo SLURM_JOB_NAME: $SLURM_JOB_NAME
echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION
echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo ------------------------------------------------------

echo Creating SCRATCH directories on nodes $SLURM_JOB_NODELIST...
SCRATCH=/scratch/$USER-$SLURM_JOB_ID
srun -n$SLURM_NNODES mkdir -m 770 -p $SCRATCH  || exit $?
echo ------------------------------------------------------
echo Transferring files from frontend to compute nodes $SLURM_JOB_NODELIST
#srun -n$SLURM_NNODES cp -rvf $SLURM_SUBMIT_DIR/* $SCRATCH  || exit $?
srun -n$SLURM_NNODES cp -rf $SLURM_SUBMIT_DIR/* $SCRATCH  || exit $?
echo ------------------------------------------------------

echo Run Python program...
module purge
cd $SCRATCH
python3 Simulation_Model.py $DATASET_NAME ${SLURM_ARRAY_TASK_ID}
echo ------------------------------------------------------

echo Transferring result files from compute nodes to frontend
#srun -n$SLURM_NNODES cp -rvf $SCRATCH  $SLURM_SUBMIT_DIR   || exit $?
srun -n$SLURM_NNODES cp -rf $SCRATCH/output/*  $SLURM_SUBMIT_DIR/output/* 2> /dev/null
echo ------------------------------------------------------
echo Deleting scratch...
#srun -n$SLURM_NNODES rm -rvf $SCRATCH  || exit 0
srun -n$SLURM_NNODES rm -rf $SCRATCH 
echo ------------------------------------------------------

