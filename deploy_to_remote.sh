#!/bin/bash

# 部署FlagScale到远程机器172.24.135.43的脚本

REMOTE_HOST="172.24.135.43"
REMOTE_USER="root"  # 根据实际情况修改用户名
REMOTE_PATH="/home/lyrawen/FlagScale"

echo "开始部署FlagScale到远程机器 $REMOTE_HOST..."

# 1. 创建远程目录
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PATH"

# 2. 同步代码到远程机器
echo "同步代码文件..."
rsync -avz --exclude='outputs/' --exclude='.git/' /home/lyrawen/FlagScale/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

# 3. 在远程机器上设置环境
echo "在远程机器上设置环境..."
ssh $REMOTE_USER@$REMOTE_HOST << 'EOF'
cd /home/lyrawen/FlagScale

# 检查conda环境是否存在
if ! conda env list | grep -q verl; then
    echo "创建verl环境..."
    conda create -n verl python=3.10 -y
fi

# 激活环境并安装依赖
source /root/miniconda3/bin/activate verl
pip install -r requirements.txt

# 设置权限
chmod +x examples/qwen2/conf/hostfile.txt
chmod +x outputs/logs/scripts/*.sh
EOF

echo "部署完成！"
echo "现在可以在远程机器上运行："
echo "ssh $REMOTE_USER@$REMOTE_HOST 'cd $REMOTE_PATH && source /root/miniconda3/bin/activate verl && python -m flagscale.runner.runner_rl --config examples/qwen2/conf/rl.yaml'"
