#!/usr/bin/env bash

ret=100
while [ $ret -eq 100 ]
do
	echo "STARTING TRAINING"
	CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --dataset_name=cnn_dm_merge --dataset_split=train --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --actual_log_root=/home/logan/data/multidoc_summarization/logs --exp_name=cnn_dm_merge --max_enc_steps=100 --max_dec_steps=30 --batch_size=4 --num_iterations=60000
	ret=$?
done
echo "SUCCESS"
