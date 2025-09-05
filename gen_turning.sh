#!/bin/sh

today=$(date +%Y-%m-%d)

nohup /home/paohe/miniconda3/envs/brain_mix/bin/python /home/paohe/Documents/Workspaces/brain-mix/nlp/datasets/2.mysql_to_turning.py > /home/paohe/Documents/Temps/brain-mix/mysql_to_turning_$today.log 2>&1 &

