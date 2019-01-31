#!/usr/bin/env bash

set -x

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
max_enc_steps=100
min_dec_steps=10
max_dec_steps=30
batch_size=128

exp_suffix=_sent_singles
dataset_split=train
num_iterations=500000
port=6006
data_root_flag=''

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
    --cuda=*)
      cuda="${1#*=}"
      ;;
    --max_enc_steps=*)
      max_enc_steps="${1#*=}"
      ;;
    --min_dec_steps=*)
      min_dec_steps="${1#*=}"
      ;;
    --max_dec_steps=*)
      max_dec_steps="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --data_root_flag=*)
      data_root_flag="${1#*=}"
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
    exp_suffix=_sent_both
elif [[ "$singles_and_pairs" = "singles" ]]; then
    exp_suffix=_sent_singles
else
    exp_suffix=""
    data_root_flag=--data_root=/home/logan/data/tf_data
fi

if [[ "$cuda" = "1" ]]; then
    port=7007
fi

echo "$dataset_name"
echo "$mode"
echo "$singles_and_pairs"
echo "$@"

if [[ "$mode" = "train" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" tensorboard --logdir=logs/"$dataset_name""$exp_suffix"/eval --port="$port" &> /home/logan/null &
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=eval --dataset_name="$dataset_name""$exp_suffix" --dataset_split=val --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations=-1 "$data_root_flag" "$@" &> /home/logan/null &
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode="$mode" --dataset_name="$dataset_name""$exp_suffix" --dataset_split="$dataset_split" --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations"  "$data_root_flag" "$@"
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode="$mode" --dataset_name="$dataset_name""$exp_suffix" --dataset_split="$dataset_split" --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations" "$data_root_flag" "$@"
elif [[ "$mode" = "train_only" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=train --dataset_name="$dataset_name""$exp_suffix" --dataset_split="$dataset_split" --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations" "$data_root_flag" "$@"
elif [[ "$mode" = "eval_only" ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=eval --dataset_name="$dataset_name""$exp_suffix" --dataset_split=val --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations=-1 "$data_root_flag" "$@"
elif [[ "$mode" = "tensorboard" ]]; then
    CUDA_VISIBLE_DEVICES=1 tensorboard --logdir=logs/"$dataset_name""$exp_suffix"/eval --port="$port"
else
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=train --dataset_name="$dataset_name""$exp_suffix" --dataset_split="$dataset_split" --exp_name="$dataset_name""$exp_suffix" --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations"  "$data_root_flag" "$@"
fi
