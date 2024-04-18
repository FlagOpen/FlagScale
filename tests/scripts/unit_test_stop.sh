pids=$(ps aux | grep pytest | grep -v grep | awk '{print $2}')

for pid in $pids
do
    kill -9 $pid
done
