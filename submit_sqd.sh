#!/bin/bash
#SBATCH --job-name=sqd-sbd
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --array=0-4
#SBATCH --output=sqd_%a_%j.out
#SBATCH --error=sqd_%a_%j.err

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

# Paths (adjust as needed)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${REPO_DIR}/../calibrate-ibm/experiments/ATP"
SBD_EXE="${REPO_DIR}/apps/chemistry_tpb_selected_basis_diagonalization/diag"

# Array index -> (type, adapt_iterations)
# 0-2: singletons for ADAPT 1, 2, 3
# 3-4: cumulatives [1,2] and [1,2,3]
case $SLURM_ARRAY_TASK_ID in
    0) TYPE="singleton"; ADAPTS="1" ;;
    1) TYPE="singleton"; ADAPTS="2" ;;
    2) TYPE="singleton"; ADAPTS="3" ;;
    3) TYPE="cumulative"; ADAPTS="1 2" ;;
    4) TYPE="cumulative"; ADAPTS="1 2 3" ;;
esac

ADAPTS_KEY=$(echo $ADAPTS | tr ' ' '_')
if [ "$TYPE" = "singleton" ]; then
    OUTPUT_DIR="${REPO_DIR}/results/singleton_${ADAPTS_KEY}"
else
    OUTPUT_DIR="${REPO_DIR}/results/cumulative_${ADAPTS_KEY}"
fi

python "${REPO_DIR}/run_sqd.py" \
    --circuit_dir "${DATA_DIR}/circuits" \
    --hamiltonian_dir "${DATA_DIR}/hamiltonians" \
    --results_dir "${DATA_DIR}/results" \
    --sbd_exe "${SBD_EXE}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_iterations 2000 \
    --resume \
    ${ADAPTS}
