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
    --cuda=*)
      cuda="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

if [[ "$mode" = "eval" ]]; then
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
    tensorboard --logdir=/home/logan/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/output "$@"
elif [[ "$mode" = "predict" ]]; then
    cd bert
    CUDA_VISIBLE_DEVICES="$cuda" python run_classifier.py   --task_name=merge   --do_predict=true   --data_dir=/home/logan/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/input  --max_seq_length=64   --output_dir=/home/logan/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/output/saved "$@"
else
    cd bert
    CUDA_VISIBLE_DEVICES="$cuda" python run_classifier.py   --task_name=merge   --do_train=true   --do_eval=true   --data_dir=/home/logan/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/input   --max_seq_length=64   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=1000.0   --output_dir=/home/logan/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/output "$@"
fi
