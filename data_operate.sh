#!/bin/sh

today=$(date +%Y-%m-%d)

py_path=/home/paohe/miniconda3/envs/brain_mix
workspace_path=/home/paohe/Documents/Workspaces/brain-mix
log_path=/home/paohe/Documents/Temps/brain-mix

# delete low quality data in elasticsearch
nohup $py_path/bin/python \
$workspace_path/nlp/datasets/3.delete_low_quality_data.py \
> $log_path/3.delete_low_quality_data_$today.log 2>&1 &

# score datas with silicon
nohup $py_path/bin/python \
$workspace_path/utils/check/check_by_file_modify.py \
$log_path/4.score_and_filter_data \
$workspace_path/nlp/datasets/4.score_and_filter_data.py \
90 \
> $log_path/4.score_and_filter_data_$today.log 2>&1 &