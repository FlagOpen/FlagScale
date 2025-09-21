
ray stop
pkill -9 -f python

sleep 5s
pkill -9 -f python


export CUDA_VISIBLE_DEVICES="0,1"
export INSTANCE_NUM=2
export SERVE_PORT=6711
export ROUTER_PORT=8012

ray start --head --port 7891


export PYTHONPATH=/mine/emu3.5/app
python run.py --config-path /mine/emu3.5 --config-name emu_config action=run
