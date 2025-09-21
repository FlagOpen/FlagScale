
ray stop
pkill -9 -f python

sleep 5s
pkill -9 -f python


export CUDA_VISIBLE_DEVICES="6,7"
#export INSTANCE_NUM=2
export SERVE_PORT=6701
export ROUTER_PORT=8012

ray start --head --port 7892


#export PYTHONPATH=/mine/emu3.5/app
python run.py --config-path /mine/dev/emu3.5 --config-name emu_howto_dev action=run
