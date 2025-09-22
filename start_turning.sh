#!/bin/sh

today=$(date +%Y-%m-%d)

py_path=/home/paohe/anaconda3/envs/brain_mix
workspace_path=/home/paohe/llm/nlp/brain-mix
log_path=/home/paohe/Documents/tmp/logs/brain-mix

# qwen3 model turning
nohup $py_path/bin/python \
$workspace_path/nlp/models/reasoning/model_auto_turning.py \
> $log_path/model_auto_turning_$today.log 2>&1 &


