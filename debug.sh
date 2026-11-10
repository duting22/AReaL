#!/bin/bash
# Launch a fresh single-node interactive debug session on Slurm.
#
# Usage:
#   ./debug.sh                      # default container image (below)
#   ./debug.sh -i IMAGE.sif         # override the container image

workdir=/storage/openpsi/users/shenxujie.sxj/
image=/storage/openpsi/images/areal-dev-sglang-20260401.sif

while getopts "i:h" opt; do
    case "$opt" in
        i) image=$OPTARG ;;
        h|*)
            echo "Usage: $0 [-i IMAGE]"
            exit 1 ;;
    esac
done
shift $((OPTIND - 1))

srun --reservation=whj --mpi=pmi2 --ntasks=1 --gres=gpu:0 --chdir="$workdir" --job-name=sxj-cpu-debug \
    --cpus-per-task=64 --mem=100G --pty \
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage "$image"
