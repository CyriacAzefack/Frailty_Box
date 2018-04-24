#!/bin/bash
#SBATCH --job-name=ARUBA_Simulation_20_rep_90_days
#SBATCH --mail-user=cyriac.azefack@emse.fr
#SBATCH --mail-type=ALL
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --exclude=compute-0-15

unset SLURM_GTIDS

DATASET_NAME=aruba

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
python3 Simulation_Model.py -n $DATASET_NAME -r ${SLURM_ARRAY_TASK_ID} --days=90 --support_min=30
echo ------------------------------------------------------

echo Transferring result files from compute nodes to frontend
#srun -n$SLURM_NNODES cp -rvf $SCRATCH  $SLURM_SUBMIT_DIR   || exit $?
srun -n$SLURM_NNODES cp -rf $SCRATCH/output/*  $SLURM_SUBMIT_DIR/output 2> /dev/null
echo ------------------------------------------------------
echo Deleting scratch...
#srun -n$SLURM_NNODES rm -rvf $SCRATCH  || exit 0
srun -n$SLURM_NNODES rm -rf $SCRATCH 
echo ------------------------------------------------------

