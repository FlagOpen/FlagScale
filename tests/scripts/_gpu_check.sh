#!/bin/bash
wait_for_gpu() {
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    local memory_usage_max=30000
    while true; do
        # Use mapfile instead of read to avoid IFS issues and ensure proper array population
        local memory_usage_array=()
        local memory_total_array=()
        # Query GPU memory usage and total memory, suppress stderr to prevent exit on failure
        mapfile -t memory_usage_array < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        mapfile -t memory_total_array < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
        local need_wait=false
        # Check each GPU's available memory
        for ((i=0; i<${#memory_usage_array[@]}; i++)); do
            # Remove whitespace from nvidia-smi output
            local memory_usage_i=${memory_usage_array[$i]// /}
            local memory_total_i=${memory_total_array[$i]// /}
            # Validate that values are numeric before calculation
            if [[ $memory_usage_i =~ ^[0-9]+$ ]] && [[ $memory_total_i =~ ^[0-9]+$ ]]; then
                local memory_remain_i=$((memory_total_i - memory_usage_i))
                # If available memory is less than required threshold, need to wait
                if [ $memory_remain_i -lt $memory_usage_max ]; then
                    need_wait=true
                    break
                fi
            else
                # Log warning for invalid values and continue waiting
                echo "Warning: Invalid memory values - usage: '$memory_usage_i', total: '$memory_total_i'"
                need_wait=true
                break
            fi
        done
        # If all GPUs have sufficient memory, break the loop
        if [ "$need_wait" = false ]; then
            break
        fi
        echo "wait for gpu free"
        sleep 1m
    done
    echo "All gpu is free"
}