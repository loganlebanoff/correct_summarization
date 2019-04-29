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
singles_and_pairs=singles
dataset_split="test val train"

cuda=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset_name=*)
      dataset_name="${1#*=}"
      ;;
    --dataset_split=*)
      dataset_split="${1#*=}"
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


echo "$dataset_name"
echo "$singles_and_pairs"
echo "$dataset_split"
echo "$@"

cd bert
for split in $dataset_split; do
    CUDA_VISIBLE_DEVICES="$cuda" python extract_features.py   --input_file=$HOME/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/input_article/"$split".tsv   --output_file=$HOME/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/output_article/"$split".jsonl   --layers=-1,-2,-3,-4   --max_seq_length=400   --batch_size=1 --only_class_embedding
done