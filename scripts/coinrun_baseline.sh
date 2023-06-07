#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job=cr-base
#SBATCH --time=3:30:00
#SBATCH --output=/cluster/home/%u/coinrun/log/baseline-32e6-%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/home/%u/coinrun/log/baseline-32e6-%j.err  # where to store error messages
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=8G


# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
# load modules
module load gcc/8.2.0 openblas/0.3.20 python/3.11.2 cuda/11.8.0 cudnn/8.8.1.3 openmpi/4.1.4 qt/5.10.0 pkg-config/0.29.2 zlib/1.2.11

cd $HOME/coinrun/coinrun
# source ../venv/bin/activate
# mpiexec -np 4 python -m coinrun.train_agent --run-id myrun

$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --restore-id baseline_long_32 --run-id baseline_long_32 --save-interval 1 -ne 64 -arch impala
for ((i=0; i < 63; i++))
do
echo "still running"
echo $i
$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id baseline_long_32 -num-eval 50 -rep 5
$HOME/coinrun/venv/bin/python3 -m coinrun.train_agent --restore-id baseline_long_32 --run-id baseline_long_32 --save-interval 1 -ne 64 -arch impala
done

$HOME/coinrun/venv/bin/python3 -m coinrun.enjoy --test-eval --restore-id baseline_long_32 -num-eval 50 -rep 5

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
