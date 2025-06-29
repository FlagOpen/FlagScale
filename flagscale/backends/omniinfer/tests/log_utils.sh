#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

# 日志颜色定义
INFO_COLOR='\033[32m'
WARNING_COLOR='\033[33m'
ERROR_COLOR='\033[31m'
RESET='\033[0m'

# 日志函数
log() {
    local level="$1"
    local message="$2"
    local datetime=$(date -u +"%Y-%m-%d %H:%M:%S" --date='+8 hours')

    case "$level" in
        INFO)
            echo -e "${INFO_COLOR}${datetime} [INFO] ${message}${RESET}"
            ;;
        WARNING)
            echo -e "${WARNING_COLOR}${datetime} [WARNING] ${message}${RESET}"
            ;;
        ERROR)
            echo -e "${ERROR_COLOR}${datetime} [ERROR] ${message}${RESET}"
            ;;
        *)
            echo "${datetime} [${level}] ${message}"
            ;;
    esac
}

# 别名函数
log_info() {
    log "INFO" "$1"
}

log_warning() {
    log "WARNING" "$1"
}

log_error() {
    log "ERROR" "$1"
}