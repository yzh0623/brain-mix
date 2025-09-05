#!/bin/sh

today=$(date +%Y-%m-%d)

# score datas
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python /home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/4.score_and_filter_data.py > /home/paohe/Documents/Temps/brain-mix/score_and_filter_data_$today.log 2>&1 &