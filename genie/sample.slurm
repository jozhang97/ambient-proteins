#!/bin/bash
#SBATCH --job-name sample_ambient_protein_diffusion

### Logging
#SBATCH --output job_logs/slurm_%j.out
#SBATCH --error  job_logs/slurm_%j.err

### Node info
#SBATCH --account CGAI24022
#SBATCH --nodes 32
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 3:00:00

nvidia-smi
hostname

source ~/.bashrc
module load gcc cuda/12.4

conda init
conda activate genie2
cd ~/code/genie3/

MODE=$1  # SHORT or LONG
EXP_NAME=$2
EPOCH=$3
SCALE=$4

if [ "$MODE" = "LONG" ]; then
  srun python genie/sample_unconditional.py --num_nodes $SLURM_NNODES --num_devices 1 \
    --min_length 299 --max_length 800 --length_step 100 \
    --batch_size 2 --num_samples 100 \
    --name $EXP_NAME --epoch $EPOCH --scale $SCALE --outdir runs/$EXP_NAME/outputs

elif [ "$MODE" = "SHORT" ]; then
  srun python genie/sample_unconditional.py --num_nodes $SLURM_NNODES --num_devices 1 \
    --min_length 49 --max_length 250 --length_step 50 --num_samples 100 \
    --name $EXP_NAME --epoch $EPOCH --scale $SCALE --outdir runs/$EXP_NAME/outputs
else
  echo "Error: MODE must be either SHORT or LONG"
  exit 1
fi
