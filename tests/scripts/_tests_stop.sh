# Before running, please ensure there are no other processes with a similar name to avoid closing the wrong one.
stop_all() {
    pids=$(ps aux | grep pytest | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done

    pids=$(ps aux | grep python | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done

    pids=$(ps aux | grep torchrun | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done
}

# Function to execute a command and clear serve process
stop_serve() {
    ray stop
    pkill -f "python"
    return 0
}