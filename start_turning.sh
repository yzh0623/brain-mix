#!/bin/sh

today=$(date +%Y_%m_%d)

py_path=/home/paohe/anaconda3/envs/brain_mix
workspace_path=/home/paohe/llm/yzh/brain-mix
log_path=/home/paohe/Documents/tmp/logs/brain-mix

# model turning script 
nohup $py_path/bin/python \
$workspace_path/nlp/models/reasoning/step1_model_auto_turning.py \
> $log_path/model_auto_turning_$today.log 2>&1 &
