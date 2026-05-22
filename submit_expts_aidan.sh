#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --mem=10g
#SBATCH --time=120:00:00
#SBATCH --array=0-999
#SBATCH -o logs/tmp_%A_%a.out
#SBATCH -e logs/tmp_%A_%a.err

source /home/higgsab/bin/myconda
conda activate rnn
 
OFFSET=0
DIR="noiseless_sigma"

while [[ $# -gt 0 ]]; do
    case $1 in
        --offset) OFFSET=$2; shift 2 ;;
        --dir)    DIR=$2;    shift 2 ;;
        *) shift ;;
    esac
done

mapfile -t CONFIGS < <(printf '%s\n' configs/${DIR}/zeroint*_delay*_normal*_sigma*.yaml \
                                     configs/${DIR}/zeroint*_delay*_vanilla*_sigma*.yaml \
                                     configs/${DIR}/zeroint*_delay*_groups*_sigma*.yaml | sort)
N_SEEDS=40

CONFIG_IDX=$(( (SLURM_ARRAY_TASK_ID + OFFSET) / N_SEEDS ))
SEED=$(( (SLURM_ARRAY_TASK_ID + OFFSET) % N_SEEDS))
CONFIG=${CONFIGS[$CONFIG_IDX]}

BASENAME=$(basename $CONFIG .yaml)
ZEROINT=$(echo $BASENAME    | grep -oP 'zeroint\K(True|False)')
DELAY=$(echo $BASENAME      | grep -oP 'delay\K[0-9]+\.?[0-9]*')
SIGMA=$(echo $BASENAME | grep -oP 'sigma\K[0-9]+\.?[0-9]*')

if echo $BASENAME | grep -q 'normal'; then
    ETA=$(echo $BASENAME | grep -oP 'eta\K[0-9]+\.?[0-9]*')
    NAME="zeroint${ZEROINT}_delay${DELAY}_normal_eta${ETA}_sigma${SIGMA}_seed${SEED}"
elif echo $BASENAME | grep -q 'groups'; then
    GROUPS_TAU=$(echo $BASENAME | grep -oP 'groups\K[0-9]+\.?[0-9]*')
    PROP=$(echo $BASENAME       | grep -oP 'prop\K[0-9]+\.?[0-9]*')
    NAME="zeroint${ZEROINT}_delay${DELAY}_groups${GROUPS_TAU}_prop${PROP}_sigma${SIGMA}_seed${SEED}"
else
    VANILLA_TAU=$(echo $BASENAME | grep -oP 'vanilla\K[0-9]+\.?[0-9]*')
    NAME="zeroint${ZEROINT}_delay${DELAY}_vanilla${VANILLA_TAU}_sigma${SIGMA}_seed${SEED}"
fi

mkdir -p logs/${DIR}
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out logs/${DIR}/${NAME}.out
mv logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err logs/${DIR}/${NAME}.err

echo "Config: $CONFIG | Seed: $SEED | Name: $NAME"
python -u run_expt.py --config=$CONFIG --seed=$SEED --name=$NAME --output_dir=$DIR

# sbatch submit_expts_aidan.sh --offset 0 --dir noiseless_sigma
# sbatch submit_expts_aidan.sh --offset 1000 --dir noiseless_sigma
# sbatch submit_expts_aidan.sh --offset 2000 --dir noiseless_sigma
# sbatch --array=0-839 submit_expts_aidan.sh --offset 3000 --dir noiseless_sigma
# i.e. for 3840 total jobs