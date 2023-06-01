#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PYTHONPATH="../:${PYTHONPATH}"

#seeds=( 606 )
seeds=( 606 1011 3004 5555 12096 )

# kill child processes on script termination
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

IFS=','

# base trainings

for SEED in "${seeds[@]}"
do
  # classification setups train
  for setup in ciSPN,NLLLoss,1e-3,true mlp,MSELoss,1e-3,true mlp,MSELoss,1e-3,false mlp,causalLoss,1e-3,true
  #for setup in mlp,MSELoss,1e-3,false
  do
    set -- $setup
    MODEL=$1
    LOSS=$2
    LR=$3
    PROVIDEINTERVENTIONS=$4

    echo "$(date +%R): SEED $SEED $setup TRAIN"
    for DATASET in CHC ASIA CANCER EARTHQUAKE CHAIN
    do
      : # empty loops are not allowed ...
      #python ./E1_tabular_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait

    echo "$(date +%R): SEED $SEED $setup EVAL"
    for DATASET in CHC ASIA CANCER EARTHQUAKE CHAIN
    do
      : # empty loops are not allowed ...
      #python ./E1_tabular_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait
  done
done

echo "base done"


LR=1e-3
MODEL=mlp
LOSS=MSELoss

# combined training
for SEED in "${seeds[@]}"
do
  for LOSS2_FACTOR in 10.0 1.0 0.1 0.01 0.001
  do
    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} TRAIN"
    for DATASET in CHC ASIA CANCER EARTHQUAKE CHAIN
    do
      : # empty loops are not allowed ...
      #python ./E1_tabular_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset $DATASET --loss2=causalLoss --loss2_factor=${LOSS2_FACTOR} &
    done
    wait

    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} EVAL"
    for DATASET in CHC ASIA CANCER EARTHQUAKE CHAIN
    do
      : # empty loops are not allowed ...
      #python ./E1_tabular_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset $DATASET --loss2=causalLoss --loss2_factor=${LOSS2_FACTOR} &
    done
    wait
  done
done

echo "combined done"


python ./E1_BN_CHC_eval.py --dataset CHC &
python ./E1_BN_eval.py --dataset ASIA &
python ./E1_BN_eval.py --dataset CANCER &
python ./E1_BN_eval.py --dataset EARTHQUAKE &
python ./E1_BN_eval.py --dataset CHAIN &
python ./E1_BN_CHC_eval.py --dataset CHC --eval_mode correlation &
python ./E1_BN_eval.py --dataset ASIA --eval_mode correlation &
python ./E1_BN_eval.py --dataset CANCER --eval_mode correlation &
python ./E1_BN_eval.py --dataset EARTHQUAKE --eval_mode correlation &
python ./E1_BN_eval.py --dataset CHAIN --eval_mode correlation &
wait

echo "BN done"
