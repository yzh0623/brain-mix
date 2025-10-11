#!/bin/sh

today=$(date +%Y_%m_%d)

py_path=/home/paohe/miniconda3/envs/brain_mix
workspace_path=/home/paohe/Documents/Workspaces/brain-mix
log_path=/home/paohe/Documents/Temps/brain-mix

# delete low quality data in elasticsearch
nohup $py_path/bin/python \
$workspace_path/nlp/datasets/step3_delete_low_quality_data.py \
> $log_path/step3_delete_low_quality_data_$today.log 2>&1 &

# score datas with silicon
nohup $py_path/bin/python \
$workspace_path/nlp/schedule/check_by_file_modify.py \
$log_path/step4_score_and_filter_data \
$workspace_path/nlp/datasets/step4_score_and_filter_data.py \
90 \
> $log_path/score_and_filter_$today.log 2>&1 &