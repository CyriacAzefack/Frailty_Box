#!/bin/bash
 
#================= OPTIONS (s'applique à chaque job du tableau) =========================
#SBATCH --job-name=job.job     					# Nom du job
#SBATCH --mail-user=cyriac.azefack@emse.fr		# Mail
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1             					# chaque job possède une seule task
#SBATCH --mem=12000            					# mémoire nécessaire (par noeud) en Mo

#========================== execution ================================
python3 Simulation_Model.py 



