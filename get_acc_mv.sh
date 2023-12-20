#! /bin/bash

exp_path=$1
result_path=result/$(echo "$exp_path" | sed 's/.*outputs\///')
echo $result_path
# mkdir $result_path
nohup unbuffer python few_shot_kbqa_mv.py -e $exp_path > $result_path 2>&1 &
