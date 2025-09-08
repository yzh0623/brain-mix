#!/bin/sh

today=$(date +%Y-%m-%d)

# sync turning datas to elasticsearch
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python \
/home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/1.load_and_save_to_es.py \
> /home/paohe/Documents/Temps/brain-mix/load_and_save_to_es_$today.log 2>&1 &

# delete low quality data in elasticsearch
nohup /home/paohe/miniconda3/envs/brain_mix/bin/python \
/home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/3.delete_low_quality_data.py \
> /home/paohe/Documents/Temps/brain-mix/delete_low_quality_data_$today.log 2>&1 &
