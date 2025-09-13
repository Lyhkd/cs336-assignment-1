#!/bin/bash
# 等待进程结束
PID=3177205
NEXT_CMD="echo '目标进程结束，开始执行后续任务' && ./next.sh"

while kill -0 "$PID" 2>/dev/null; do
    # kill -0 不会真的杀进程，只是检查 PID 是否存在
    sleep 5
done
# 进程结束后执行另一个命令
echo "训练结束，开始评估"
python -m cs336_basics.train --config config/lr7e3-7e4-bs256.json