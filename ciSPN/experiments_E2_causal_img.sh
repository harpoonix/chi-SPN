#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PYTHONPATH="../:${PYTHONPATH}"

seeds=( 606 1011 3004 5555 12096 )

# kill child processes on script termination
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

IFS=','

# base trainings

LR=1e-4
for SEED in "${seeds[@]}"
do
  echo "$(date +%R): SEED $SEED A"
  #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model cnn --loss MSELoss --dataset hiddenObject --provide_interventions false &
  #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model cnn --loss MSELoss --dataset hiddenObject --provide_interventions true &
  #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model ciCNNSPN --loss NLLLoss --dataset hiddenObject --provide_interventions true &

  wait

  echo "$(date +%R): SEED $SEED B"
  #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model cnn --loss causalLoss --dataset hiddenObject --provide_interventions true &
  #python ./E2_causal_img_eval.py --seed $SEED --model cnn --loss MSELoss --dataset hiddenObject --provide_interventions false &
  #python ./E2_causal_img_eval.py --seed $SEED --model cnn --loss MSELoss --dataset hiddenObject --provide_interventions true &
  wait

  echo "$(date +%R): SEED $SEED C"
  #python ./E2_causal_img_eval.py --seed $SEED --model ciCNNSPN --loss NLLLoss --dataset hiddenObject --provide_interventions true &
  #python ./E2_causal_img_eval.py --seed $SEED --model cnn --loss causalLoss --dataset hiddenObject --provide_interventions true &
  wait
done

echo "base done"


LR=1e-4
MODEL=cnn
LOSS=MSELoss

# combined training
for SEED in "${seeds[@]}"
do
    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} A"
    #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=10.0 &
    #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=1.0 &
    wait

    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} B"
    #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.1 &
    #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.01 &
    wait

    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} C"
    #python ./E2_causal_img_train.py --seed $SEED --lr $LR --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.001 &
    #python ./E2_causal_img_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=10.0 &
    wait

    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} D"
    #python ./E2_causal_img_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=1.0 &
    #python ./E2_causal_img_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.1 &
    wait

    echo "$(date +%R): SEED $SEED ${LOSS2_FACTOR} E"
    #python ./E2_causal_img_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.01 &
    #python ./E2_causal_img_eval.py --seed $SEED --model $MODEL --loss $LOSS --dataset hiddenObject --loss2=causalLoss --loss2_factor=0.001 &
    wait
done

echo "combined done"


for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  echo "ID $ID"
  #python ./E2_causal_img_heatmap.py --seed 606 --model ciCNNSPN --loss NLLLoss --dataset hiddenObject --sample_id $ID --provide_interventions true --inspect_output 2 &
  #python ./E2_causal_img_heatmap.py --seed 606 --model cnn --loss MSELoss --dataset hiddenObject --sample_id $ID --provide_interventions true --inspect_output 2 &
  #python ./E2_causal_img_heatmap.py --seed 606 --model cnn --loss MSELoss --dataset hiddenObject --sample_id $ID --provide_interventions false  --inspect_output 2 &
  #python ./E2_causal_img_heatmap.py --seed 606 --model cnn --loss causalLoss --dataset hiddenObject --sample_id $ID --provide_interventions true --inspect_output 2 &
  wait
done

echo "heatmaps done"