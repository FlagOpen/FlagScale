#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

# 检查参数数量
if [ $# -ne 3 ]; then
    log_error "Usage: $0 <IMAGE_NAME> <PLAYBOOK_YML> <INVENTORY_YML>"
    exit 1
fi

IMAGE_NAME=${1}
PLAYBOOK_YML=${2}
INVENTORY_YML=${3}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH
source "$SCRIPT_PATH/../../log_utils.sh"

log_info "IMAGE_NAME 的值: ${IMAGE_NAME}"
log_info "PLAYBOOK_YML 的值: ${PLAYBOOK_YML}"
log_info "INVENTORY_YML 的值: ${INVENTORY_YML}"

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: File $1 not found"
        exit 1
    fi
}

check_file "$PLAYBOOK_YML"
check_file "$INVENTORY_YML"


# 替换函数
update_yml() {
    local file="$1"
    local key="DOCKER_IMAGE_ID"

    # 使用sed进行原地替换
    sed -i "s#^\([[:space:]]*${key}:[[:space:]]*\)\"[^\"]*\"#\1\"$IMAGE_NAME\"#" "$file"

    # 验证替换结果
    if grep -q "${key}:[[:space:]]*\"$IMAGE_NAME\"" "$file"; then
        log_info "Successfully updated $key to $IMAGE_NAME in $file"
    else
        log_error "Error: Failed to update $key in $file"
        exit 1
    fi

}

# 更新文件
update_yml "$PLAYBOOK_YML"

log_info "执行ansible-playbook命令"
ansible-playbook -i ${INVENTORY_YML} ${PLAYBOOK_YML}

exit 0
