#!/bin/bash

#===============================================================================
# exemples d'options


#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=1     # 1 seul CPU pour une task
#SBATCH --mem=2000            # m�moire n�cessaire (par noeud) en Mo

#===============================================================================
#ex�cution du programme 
python3 xED_algorithm.py