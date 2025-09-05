#!/bin/sh

today=$(date +%Y-%m-%d)

# generate turning datas
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python /home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/2.mysql_to_turning.py > /home/paohe/Documents/Temps/brain-mix/mysql_to_turning_$today.log 2>&1 &

# delete low quality data
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python /home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/3.delete_low_quality_data.py > /home/paohe/Documents/Temps/brain-mix/delete_low_quality_data_$today.log 2>&1 &
