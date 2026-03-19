#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<youremail>@nyu.edu

set -e

export SINGULARITYENV_WANDB_INSECURE_DISABLE_SSL=true

singularity exec --nv --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3:ro /scratch/<youremail>/SIF/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
/bin/bash << 'EOF'

# This is a dummy setting to reach your code directory
cd /scratch/<yourfolder>/EMERGE/PufferDrive
source .venv/bin/activate
export WANDB_INSECURE_DISABLE_SSL=true
python setup.py build_ext --inplace --force
bash scripts/build_ocean.sh visualize local

# Start the GPU heartbeat in the background with lower priority
nice -n 19 python gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU Heartbeat with PID: $HEARTBEAT_PID"

# Run your main workload
puffer train puffer_drive --wandb --wandb-project "pufferproject" --wandb-group "test"

# Clean up: kill the heartbeat process when done
kill $HEARTBEAT_PID 2>/dev/null || true
echo "Stopped GPU Heartbeat"

EOF
