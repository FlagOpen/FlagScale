#!/bin/bash
NGINX_SBIN_PATH="${NGINX_SBIN_PATH:-/usr/local/nginx}"
export PATH=${NGINX_SBIN_PATH}:${PATH}

if grep -qaE 'docker|kubepods|containerd' /proc/1/cgroup || [ -f /.dockerenv ]; then
    IN_CONTAINER=true
else
    IN_CONTAINER=false
fi

######################
## os configuration ##
######################

function set_limits_conf() {
    \cp -n /etc/security/limits.conf /etc/security/limits.conf_bak
    echo "*	soft	nofile	102400" >> /etc/security/limits.conf
    echo "*	hard	nofile	102400" >> /etc/security/limits.conf
}

function set_sysctl_conf() {
    \cp -n /etc/sysctl.conf /etc/sysctl.conf_bak
    echo "net.ipv4.tcp_tw_reuse = 1" >> /etc/sysctl.conf
    echo "net.ipv4.tcp_keepalive_time = 60" >> /etc/sysctl.conf
    echo "net.ipv4.tcp_fin_timeout = 1" >> /etc/sysctl.conf
    echo "net.ipv4.tcp_max_tw_buckets = 5000" >> /etc/sysctl.conf
    echo "net.ipv4.ip_local_port_range = 1024     65500" >> /etc/sysctl.conf
    echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
    echo "net.ipv4.tcp_max_syn_backlog = 262144" >> /etc/sysctl.conf
    echo "net.core.netdev_max_backlog = 262144" >> /etc/sysctl.conf
    echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
    echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
    echo "net.netfilter.nf_conntrack_max = 0" >> /etc/sysctl.conf
    echo "net.nf_conntrack_max = 0" >> /etc/sysctl.conf
    /sbin/sysctl -p
}

function set_selinux() {
    \cp -n /etc/sysconfig/selinux /etc/sysconfig/selinux_bak
    sed -i 's/SELINUX=enforcing/SELINUX=disabled/g' /etc/sysconfig/selinux
}

function os_configuration() {
    set_limits_conf
    set_sysctl_conf
    set_selinux
}

function rollback_os_config() {
    if [[ -f "/etc/security/limits.conf_bak" ]]; then
        \cp /etc/security/limits.conf_bak /etc/security/limits.conf
    fi

    if [[ -f "/etc/sysctl.conf_bak" ]]; then
        \cp /etc/sysctl.conf_bak /etc/sysctl.conf
    fi

    if [[ -f "/etc/sysconfig/selinux_bak" ]]; then
        \cp /etc/sysconfig/selinux_bak /etc/sysconfig/selinux
    fi
}

#########################
## nginx configuration ##
#########################

function create_default_nginx_conf() {
    local nginx_conf_file="$1"

    cat <<EOF > $nginx_conf_file

worker_processes  1;

events {
    worker_connections  1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;

    sendfile        on;
    keepalive_timeout  65;

    server {
        listen       80;
        server_name  localhost;

        location / {
            root   html;
            index  index.html index.htm;
        }

        #error_page  404              /404.html;

        # redirect server error pages to the static page /50x.html
        #
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
    }
}
EOF
}

function nginx_set_worker_processes() {
    local nginx_conf_file="$1"
    local core_num="$2"

    if [[ ! -f "$nginx_conf_file" ]]; then
        echo "Error: nginx conf file '$nginx_conf_file' does not exist."
        return 1
    fi

    # Set or update worker_processes
    if grep -q "^[[:space:]]*worker_processes" "$nginx_conf_file"; then
        # Replace the existing worker_processes line
        sed -i "s|^[[:space:]]*worker_processes[[:space:]].*;|worker_processes ${core_num};|" "$nginx_conf_file"
    else
        # Add worker_processes at the top of the file if not present
        sed -i "1i worker_processes ${core_num};" "$nginx_conf_file"
    fi
}

function nginx_worker_cpu_affinity_layout() {
    local start_core_index="$1"
    local core_num="$2"
    local total_cores=$((start_core_index + core_num))
    local layout=""
    for ((i=0; i<core_num; i++)); do
        mask=""
        for ((j=0; j<total_cores; j++)); do
            if (( j == start_core_index + i )); then
                mask="1${mask}"
                break
            else
                mask="0${mask}"
            fi
        done
        layout="${layout}${mask} "
    done
}

function nginx_set_worker_cpu_affinity() {
    local nginx_conf_file="$1"
    local start_core_index="$2"
    local core_num="$3"

    if [[ ! -f "$nginx_conf_file" ]]; then
        echo "Error: nginx conf file '$nginx_conf_file' does not exist."
        return 1
    fi

    local affinity_line="worker_cpu_affinity $(nginx_worker_cpu_affinity_layout "$start_core_index" "$core_num");"

    # Check if worker_cpu_affinity already exists
    if grep -q "^[[:space:]]*worker_cpu_affinity" "$nginx_conf_file"; then
        # Replace the existing line
        sed -i "s|^[[:space:]]*worker_cpu_affinity.*|${affinity_line}|" "$nginx_conf_file"
    else
        # Add after worker_processes directive
        if grep -q "^[[:space:]]*worker_processes" "$nginx_conf_file"; then
            sed -i "/^[[:space:]]*worker_processes.*/a ${affinity_line}" "$nginx_conf_file"
        else
            # Add at the top if worker_processes is not found
            sed -i "1i ${affinity_line}" "$nginx_conf_file"
        fi
    fi
}

function nginx_set_error_log() {
    local nginx_conf_file="$1"
    local log_file="$2"
    local log_level="$3"
    local error_log_line="error_log $log_file $log_level;"

    if [[ -z $log_file ]]; then
        return
    fi

    # Check if log_file's directory exists, create it if not
    log_dir=$(dirname "$log_file")
    if [[ ! -d "$log_dir" ]]; then
        mkdir -p "$log_dir"
    fi

    sed -i "1i ${error_log_line}" "$nginx_conf_file"
}

function nginx_set_worker_rlimit_nofile() {
    local nginx_conf_file="$1"
    local worker_rlimit_nofile_line="worker_rlimit_nofile 102400;"

    if [[ ! -f "$nginx_conf_file" ]]; then
        echo "Error: nginx conf file '$nginx_conf_file' does not exist."
        return 1
    fi

    # Check if worker_cpu_affinity already exists
    if grep -q "^[[:space:]]*worker_rlimit_nofile" "$nginx_conf_file"; then
        # Replace the existing line
        sed -i "s|^[[:space:]]*worker_rlimit_nofile.*|${worker_rlimit_nofile_line}|" "$nginx_conf_file"
    else
        # Add after worker_processes directive
        if grep -q "^[[:space:]]*worker_processes" "$nginx_conf_file"; then
            sed -i "/^[[:space:]]*worker_processes.*/a ${worker_rlimit_nofile_line}" "$nginx_conf_file"
        else
            # Add at the top if worker_processes is not found
            sed -i "1i ${worker_rlimit_nofile_line}" "$nginx_conf_file"
        fi
    fi
}

function nginx_set_events_config_with_sed() {
    local nginx_conf_file="$1"
    local keyword="$2"
    local line="$3"

    if grep -q "^[[:space:]]*${keyword}" "$nginx_conf_file"; then
        # Replace the existing line
        sed -i "s|^[[:space:]]*${keyword}.*|    ${line}|" "$nginx_conf_file"
    else
        # Add after events directive
        if grep -q "^[[:space:]]*events" "$nginx_conf_file"; then
            sed -i "/^[[:space:]]*events.*/a \    ${line}" "$nginx_conf_file"
        fi
    fi
}

function nginx_set_events_config() {
    local nginx_conf_file="$1"
    local worker_connections_line="worker_connections 102400;"
    local multi_accept_on_line="multi_accept on;"
    local accept_mutex_off_line="accept_mutex off;"
    local use_epoll_line="use epoll;"

    if [[ ! -f "$nginx_conf_file" ]]; then
        echo "Error: nginx conf file '$nginx_conf_file' does not exist."
        return 1
    fi

    nginx_set_events_config_with_sed $nginx_conf_file "worker_connections" "$worker_connections_line"
    nginx_set_events_config_with_sed $nginx_conf_file "multi_accept" "$multi_accept_on_line"
    nginx_set_events_config_with_sed $nginx_conf_file "accept_mutex" "$accept_mutex_off_line"
    nginx_set_events_config_with_sed $nginx_conf_file "epoll" "$use_epoll_line"
}

get_first_word() {
    local input="$1"
    echo "${input%% *}"
}

function nginx_set_http_config_with_sed() {
    local nginx_conf_file="$1"
    local line="$2"
    if [ -n "$3" ]; then
        local keyword="$3"
    else
        local keyword=$(get_first_word $line)
    fi


    if grep -q "^[[:space:]]*${keyword}" "$nginx_conf_file"; then
        # Replace the existing line
        sed -i "s|^[[:space:]]*${keyword}.*|    ${line}|" "$nginx_conf_file"
    else
        # Add after events directive
        if grep -q "^[[:space:]]*http[[:space:]]*{" "$nginx_conf_file"; then
            sed -i "/^[[:space:]]*http[[:space:]]*{.*/a \    ${line}" "$nginx_conf_file"
        fi
    fi
}

function nginx_set_http_config() {
    local nginx_conf_file="$1"
    local tcp_no_push_on_line="tcp_nopush on;"
    local tcp_no_delay_on_line="tcp_nodelay	on;"
    local send_file_max_chunk_line="sendfile_max_chunk 512k;"
    local keep_alive_requests_line="keepalive_requests 2000;"
    local client_header_buffer_size_line="client_header_buffer_size 512k;"
    local large_client_header_buffers_line="large_client_header_buffers 4 512k;"
    local client_body_buffer_size_line="client_body_buffer_size 128K;"
    local client_max_body_size_line="client_max_body_size 100m;"
    local proxy_read_timeout_line="proxy_read_timeout 600s;"
    local subrequest_output_buffer_size_line="subrequest_output_buffer_size 1m;"

    if [[ ! -f "$nginx_conf_file" ]]; then
        echo "Error: nginx conf file '$nginx_conf_file' does not exist."
        return 1
    fi

    nginx_set_http_config_with_sed $nginx_conf_file "$tcp_no_push_on_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$tcp_no_delay_on_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$send_file_max_chunk_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$keep_alive_requests_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$client_header_buffer_size_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$large_client_header_buffers_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$client_body_buffer_size_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$client_max_body_size_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$proxy_read_timeout_line"
    nginx_set_http_config_with_sed $nginx_conf_file "$subrequest_output_buffer_size_line"
}

function nginx_set_listen_port() {
    local nginx_conf_file="$1"
    local listen_port="$2"

    sed -i -E "s/(listen[[:space:]]+)[0-9]+/\1$listen_port/g" $nginx_conf_file
}

function nginx_set_reuseport() {
    local nginx_conf_file="$1"

    sed -i -E '/listen[[:space:]]+[^;]*reuseport/! s/(listen[[:space:]]+[^;]+);/\1 reuseport;/' $nginx_conf_file
}

function nginx_set_upstream() {
    local nginx_conf_file="$1"
    local servers_list="$2"
    local upstream_name="$3"

    # Build server lines with 8 spaces indentation
    local upstream_servers=""
    IFS=',' read -ra ADDR <<< "$servers_list"
    for srv in "${ADDR[@]}"; do
        upstream_servers+="        server $srv max_fails=3 fail_timeout=10s;\n"
    done

    # Compose new upstream block with 8 spaces indentation
    local upstream_block="    upstream $upstream_name {
        #length_balance;
        zone backend 64k;
        least_conn;
        keepalive 32;
${upstream_servers}
    }"

    # Remove existing upstream block (simple, not foolproof for nested/complex configs)
    awk -v name="$upstream_name" '
    BEGIN {in_block=0}
    $1=="upstream" && $2==name "{" {in_block=1}
    in_block && /\{/ {depth=1; next}
    in_block && /\}/ {depth--; if(depth==0) {in_block=0; next}}
    in_block {next}
    {print}
    ' "$nginx_conf_file" > "${nginx_conf_file}.tmp"

    # Insert new upstream block after first 'http {' line
    awk -v block="$upstream_block" '
    /http[[:space:]]*\{/ && !x {
        print
        print block "\n"
        x=1
        next
    }
    {print}
    ' "${nginx_conf_file}.tmp" > "${nginx_conf_file}.tmp2"

    # Move back
    mv "${nginx_conf_file}.tmp2" "$nginx_conf_file"
    rm -f "${nginx_conf_file}.tmp"
}

function nginx_set_location_openai_compatible() {
    local nginx_conf_file="$1"

    local location_block="
        # match all API of v1
        location /v1 {
            proxy_pass http://prefill_servers;
            proxy_http_version 1.1;
            proxy_set_header Connection "Keep-Alive";
        }

        # match /v1/completions and /v1/chat/completions
        location ~ ^/v1(/chat)?/completions$ {
            prefill /prefill_internal;
            proxy_pass http://decode_servers;
            proxy_http_version 1.1;
            proxy_set_header Connection "Keep-Alive";
        }

        # match /prefill_internal for internal prefill subrequest
        location /prefill_internal {
            internal;
            rewrite /prefill_internal/(.*) /\$1 break;
            proxy_pass http://prefill_servers;
            proxy_http_version 1.1;
            proxy_set_header Connection "Keep-Alive";
        }"
    awk -v block="$location_block" '
    BEGIN { in_server=0; brace_depth=0; inserted=0 }
    {
        # Detect entering server block
        if ($0 ~ /^[[:space:]]*server[[:space:]]*{/) {
            in_server=1
            brace_depth=1
            print
            next
        }
        # Inside server block
        if (in_server) {
            # Count braces to track nesting
            brace_depth += gsub(/{/, "{")
            brace_depth -= gsub(/}/, "}")
            # If at top level and see closing }, insert block before it
            if (brace_depth == 0 && !inserted) {
                print block
                inserted=1
                in_server=0
            }
            print
            next
        }
        print
    }
    ' "${nginx_conf_file}" > "${nginx_conf_file}.tmp"

    # Move back
    mv "${nginx_conf_file}.tmp" "$nginx_conf_file"
}

function nginx_set_load_modules() {
    local nginx_conf_file="$1"
    local load_module_set_request_id_line="load_module /usr/local/nginx/modules/ngx_http_set_request_id_module.so;"
    local load_module_prefill_line="load_module /usr/local/nginx/modules/ngx_http_prefill_module.so;"
    local load_module_upstream_length_balance_line="load_module /usr/local/nginx/modules/ngx_http_upstream_length_balance_module.so;"

    # Add all load module at the top
    sed -i "1i ${load_module_set_request_id_line}" "$nginx_conf_file"
    sed -i "2i ${load_module_prefill_line}" "$nginx_conf_file"
    sed -i "3i ${load_module_upstream_length_balance_line}" "$nginx_conf_file"

}

function nginx_configuration() {
    local nginx_conf_file="$1"
    local start_core_index="$2"
    local core_num="$3"
    local listen_port="$4"
    local prefill_servers_list="$5"
    local decode_servers_list="$6"
    local log_file="$7"
    local log_level="$8"

    \cp -n $nginx_conf_file "$nginx_conf_file"_bak
    create_default_nginx_conf $nginx_conf_file
    nginx_set_worker_processes $nginx_conf_file $core_num
    # nginx_set_worker_cpu_affinity $nginx_conf_file $start_core_index $core_num
    nginx_set_worker_rlimit_nofile $nginx_conf_file
    nginx_set_error_log $nginx_conf_file $log_file $log_level
    nginx_set_events_config $nginx_conf_file
    nginx_set_http_config $nginx_conf_file
    nginx_set_listen_port $nginx_conf_file $listen_port
    nginx_set_reuseport $nginx_conf_file
    nginx_set_upstream $nginx_conf_file $decode_servers_list "decode_servers"
    nginx_set_upstream $nginx_conf_file $prefill_servers_list "prefill_servers"
    nginx_set_location_openai_compatible $nginx_conf_file
    nginx_set_load_modules $nginx_conf_file
}

function rollback_nginx_config() {
    local nginx_conf_file="$1"
    local backup_file="$nginx_conf_file"_bak

    if [[ -f "$backup_file" ]]; then
        \cp "$backup_file" $nginx_conf_file
    fi
}

function stop_global_proxy() {
    # Wait for the processes to stop
    while pgrep nginx > /dev/null; do
        echo "Stopping existing nginx ..."
        pgrep nginx | xargs kill -15
        sleep 1
    done
    echo "Global proxy is stopped."
}

function start_global_proxy() {
    local nginx_conf_file="$1"

    # Check if config is valid
    nginx -t -c "$nginx_conf_file"
    if [ $? -ne 0 ]; then
        echo "Error: nginx config $nginx_conf_file is invalid. Exiting."
        exit 1
    fi

    echo "Starting nginx with config $nginx_conf_file..."
    nginx -c "$nginx_conf_file"
}

nginx_conf_file="/usr/local/nginx/conf/nginx.conf"
start_core_index="0"
core_num="16"
listen_port="8080"
prefill_servers_list=""
decode_servers_list=""
stop=false
rollback=false
log_file=""
log_level=""

print_help() {
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --nginx-conf-file <path>, -f <path>        Path to nginx config file"
    echo "  --start-core-index <N>,  -s <N>            Starting index of CPU core"
    echo "  --core-num <N>,         -c <N>             Number of CPU cores to use"
    echo "  --listen-port <PORT>,   -p <PORT>          Listening port"
    echo "  --prefill-servers-list <list>              Comma-separated backend servers for prefill (required to start proxy)"
    echo "  --decode-servers-list <list>               Comma-separated backend servers for decode (required to start proxy)"
    echo "  --log-file <path>,      -l <path>          Log file path"
    echo "  --log-level <LEVEL>                        Log level (e.g. debug, info, notice, warn, error, crit, alert, emerg)"
    echo "  --stop,                -S                  Stop global proxy"
    echo "  --rollback,            -R                  Rollback configuration when stopping"
    echo "  --help,                -h                  Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  Start global proxy:"
    echo "    $0 --listen-port 8080 \\"
    echo "       --prefill-servers-list 127.0.0.1:8001,127.0.0.1:8002 \\"
    echo "       --decode-servers-list 127.0.0.1:9001,127.0.0.1:9002 \\"
    echo "       --log-file /var/log/proxy.log --log-level info"
    echo ""
    echo "  Stop global proxy:"
    echo "    $0 -S"
    echo ""
    echo "  Stop global proxy with rollback:"
    echo "    $0 -S -R"
}

# Parse arguments in --xxx and -x format
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nginx-conf-file|-f)
            nginx_conf_file="$2"
            shift 2
            ;;
        --start-core-index|-s)
            start_core_index="$2"
            shift 2
            ;;
        --core-num|-c)
            core_num="$2"
            shift 2
            ;;
        --listen-port|-p)
            listen_port="$2"
            shift 2
            ;;
        --prefill-servers-list)
            prefill_servers_list="$2"
            shift 2
            ;;
        --decode-servers-list)
            decode_servers_list="$2"
            shift 2
            ;;
        --log-file|-l)
            log_file="$2"
            shift 2
            ;;
        --log-level)
            log_level="$2"
            shift 2
            ;;
        --stop|-S)
            stop=true
            shift 1
            ;;
        --rollback|-R)
            rollback=true
            shift 1
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

if [ "$stop" = false ]; then
    if [[ -z "$prefill_servers_list" || -z "$decode_servers_list" ]]; then
        echo "Missing required arguments."
        echo "Use --help for usage."
        exit 1
    fi
fi

if ! [[ "$listen_port" =~ ^[0-9]+$ ]] || [[ "$listen_port" -lt 1024 || "$listen_port" -gt 65500 ]]; then
    echo "Error: --listen-port/-p must be an integer in the range [1024, 65500]."
    exit 1
fi

function do_start() {
    nginx_configuration "$nginx_conf_file" "$start_core_index" "$core_num" "$listen_port" "$prefill_servers_list" "$decode_servers_list" "$log_file" "$log_level"
    if [ "$IN_CONTAINER" = false ]; then
        os_configuration
    fi
    stop_global_proxy
    start_global_proxy "$nginx_conf_file"
}

function do_rollback() {
    rollback_nginx_config "$nginx_conf_file"

    if [ "$IN_CONTAINER" = false ]; then
        rollback_os_config
    fi
}

function do_stop() {
    stop_global_proxy
    if [ "$rollback" = true ]; then
        do_rollback
    fi
}

function main() {
    if [ "$stop" = false ]; then
        do_start
    else
        do_stop
    fi
}

main