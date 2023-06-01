#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export PYTHONPATH="../:${PYTHONPATH}"

seeds=( 606 1011 3004 5555 12096 )

# Run all experiments and evaluations for the paper
# Uncomment the respective lines to run only a partial set of configurations

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

IFS=','

# base dt trainings
for SEED in "${seeds[@]}"
do
  for setup in CausalLossScore,true GiniIndex,true GiniIndex,false
  do
    set -- $setup
    SCORE=$1
    PROVIDEINTERVENTIONS=$2

    echo "$(date +%R): DT SEED $SEED $SCORE TRAIN"
    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_train.py --model DT --score $SCORE --seed $SEED --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait

    echo "$(date +%R): DT SEED $SEED $SCORE EVAL"
    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_eval.py --model DT --score $SCORE --seed $SEED --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
      #python ./E3_DT_eval.py --model DT --score $SCORE --seed $SEED --dataset $DATASET --do_eval=False --provide_interventions $PROVIDEINTERVENTIONS &
      # --do_eval=False # only replot trees
    done
    wait
  done

  echo "$(date +%R): DT $SEED SCIKIT TRAIN"
  for setup in GiniIndex,true GiniIndex,false
  do
    set -- $setup
    SCORE=$1
    PROVIDEINTERVENTIONS=$2

    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_scikit_train.py --model DTSciKit --score $SCORE --seed $SEED --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait

    echo "$(date +%R): DT $SEED SCIKIT EVAL"
    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_eval.py --model DTSciKit --score $SCORE --seed $SEED --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
      #python ./E3_DT_eval.py --model DTSciKit --score $SCORE --seed $SEED --dataset $DATASET --do_eval=False --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait
  done
done

echo "base done"


# combined score dts
for SEED in "${seeds[@]}"
do
  for SCORE_FACTOR in 10 1.0 0.1 0.01 0.001
  do
    echo "$(date +%R): DT combined SEED $SEED ${SCORE_FACTOR} TRAIN"
    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_train.py --model DT --score GICL --score_alpha $SCORE_FACTOR --seed $SEED --dataset $DATASET --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait

    echo "$(date +%R): DT combined SEED $SEED ${SCORE_FACTOR} EVAL"
    for DATASET in ASIA CANCER EARTHQUAKE CHAIN
    do
      #: # empty loops are not allowed ...
      python ./E3_DT_eval.py --model DT --score GICL --score_alpha $SCORE_FACTOR --seed $SEED --dataset $DATASET --do_plot False --provide_interventions $PROVIDEINTERVENTIONS &
    done
    wait
  done
done

echo "done"
