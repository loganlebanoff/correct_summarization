#!/usr/bin/env bash
intexit() {
    # Kill all subprocesses (all processes in the current process group)
    kill -HUP -$$
}

hupexit() {
    # HUP'd (probably by intexit)
    echo
    echo "Interrupted"
    exit
}

trap hupexit HUP
trap intexit INT


dataset_name=cnn_dm
mode=train
singles_and_pairs=singles

cuda=0
exp_suffix=_singles
dataset_split=train
num_iterations=500000

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset_name=*)
      dataset_name="${1#*=}"
      ;;
    --mode=*)
      mode="${1#*=}"
      ;;
    --singles_and_pairs=*)
      singles_and_pairs="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

if [[ "$mode" = "eval" ]]; then
    cuda=1
    dataset_split=val
    num_iterations=-1
fi

if [[ "$mode" = "tensorboard" ]]; then
    cuda=1
    dataset_split=val
    num_iterations=-1
fi

if [[ "$singles_and_pairs" = "both" ]]; then
    exp_suffix=""
fi

echo "$dataset_name"
echo "$mode"
echo "$singles_and_pairs"
echo "$@"

if [[ "$mode" = "tensorboard" ]]; then
    tensorboard --logdir=logs/"$dataset_name"_sent"$exp_suffix"/eval "$@"
else
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode="$mode" --dataset_name="$dataset_name"_sent"$exp_suffix" --dataset_split="$dataset_split" --exp_name="$dataset_name"_sent"$exp_suffix" --max_enc_steps=100 --min_dec_steps=10 --max_dec_steps=30 --single_pass=False --batch_size=128 --num_iterations="$num_iterations"  "$@"
fi
