#!/bin/bash

# 检查是否提供了 hostfile 参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi

hostfile="$1"

# 检查 hostfile 文件是否存在
if [ ! -f "$hostfile" ]; then
    echo "Error: Hostfile '$hostfile' not found."
    exit 1
fi

# 读取 hostfile 文件中的机器 IP
while IFS= read -r line; do
    # 跳过空行和注释行
    if [[ -z "$line" ]] || [[ "$line" =~ ^# ]]; then
        continue
    fi

    # 提取 IP 地址
    ip=$(cut -d' ' -f1 <<< "$line")

    # 在远程机器上执行 nvidia-smi 并输出结果
    echo "===== $ip ====="
    ssh -p 6700 "$ip" "nvidia-smi" << EOF
exit
EOF
    echo ""  # 添加空行以分隔不同机器的输出
done < "$hostfile"
