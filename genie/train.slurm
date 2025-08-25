#!/bin/bash
#SBATCH --job-name train_ambient_protein_diffusion

### Logging
#SBATCH --output job_logs/slurm_%j.out
#SBATCH --error  job_logs/slurm_%j.err

### Node info
#SBATCH --account CGAI24022
#SBATCH --nodes 48
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00

nvidia-smi
hostname

source ~/.bashrc
module load gcc cuda/12.4

conda init
conda activate genie2
cd ~/code/genie3/

echo "Running command: srun python -m genie.train --devices 1 --num_nodes $SLURM_NNODES --config $1"
srun python -m genie.train --devices 1 --num_nodes $SLURM_NNODES --config $1
