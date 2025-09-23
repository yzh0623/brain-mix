#!/bin/sh

today=$(date +%Y-%m-%d)

py_path=/home/yzh/anaconda3/envs/brain_mix
workspace_path=/home/yzh/llm/nlp/brain-mix
log_path=/home/yzh/Documents/tmp/logs/brain-mix

# model turning script 
nohup $py_path/bin/python \
$workspace_path/nlp/models/reasoning/model_auto_turning.py \
> $log_path/model_auto_turning_$today.log 2>&1 &


