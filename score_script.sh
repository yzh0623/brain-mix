#!/bin/sh

today=$(date +%Y-%m-%d)

# score datas with check
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python \
/home/paohe/Documents/Workspaces/brain-mix/utils/check/check_by_file_modify.py \
/home/paohe/Documents/Temps/brain-mix/4.score_and_filter_data \
/home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/4.score_and_filter_data.py \
90 \
> /home/paohe/Documents/Temps/brain-mix/check_by_file_modify_$today.log 2>&1 &