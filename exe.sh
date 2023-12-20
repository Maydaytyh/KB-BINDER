#!/bin/bash

# 设置要监控的进程 PID
process_pid="215267"

# 设置要执行的代码段
# code_to_execute="python get_prompt.py"
# code_to_execute_1 = "python split.py"
# code_to_execute="echo 'hello world'"
# 循环检查进程是否存在
while true; do
    # 检查 /proc 文件系统中的进程目录是否存在
    if [ -d "/proc/$process_pid" ]; then
        # 进程存在，等待一段时间后重新检查
        sleep 1
    else
        # 进程不存在，执行指定的代码段并退出循环
        echo "Process with PID $process_pid has finished. Executing the specified code."
        echo "Process with PID $process_pid has finished. Executing the specified code."
        nohup unbuffer python few_shot_kbqa_1.py -e outputs/grailqa_dev_expression_dev_1000_beam_search_20231205_2hop_top_4_10examples_new_rank_test_ori_llama.json > result/grailqa_dev_expresion_dev_1000_beam_search_20231205_2hop_top_4_10examples_new_rank_test_ori_llama.json 2>&1 &
        break
    fi
done
