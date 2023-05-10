#!/bin/bash

#SBATCH --job=coinrun-eps
#SBATCH --output=log/eps-%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=log/eps-%j.err  # where to store error messages
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_2080_ti:1
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

cd $HOME/coinrun/coinrun
$HOME/venv/bin/python3 -m coinrun.train_agent --run-id eps --num-levels 500 --epsilon-greedy 0.05

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
