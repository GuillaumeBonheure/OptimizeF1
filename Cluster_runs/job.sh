#!/bin/bash
#SBATCH -n 24
#SBATCH -t 8-00:00
#SBATCH -p sched_mit_sloan_interactive
#SBATCH --mem 512000
#SBATCH -o /home/gbonheur/out
#SBATCH -e /home/gbonheur/err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gbonheur@mit.edu

module load julia/1.7.3
module load gurobi/8.1.1

grbgetkey 79bb00b6-6c13-11ed-bece-0242ac120003
export GRB_LICENSE_FILE=/home/gbonheur/gurobi.lic

# races TH DH threads hours
julia cluster.jl 25 2 6 24 8
