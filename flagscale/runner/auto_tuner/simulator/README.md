# Environment
Begin at the root path of `FlagScale` repository:
```
cd flagscale/flagscale/runner/auto_tuner/simulator/custom_backend/
python setup.py develop
```

# Setup
Set necessary parameters in `config_gen.py`. For example:
```
device_type_list = ["A", "B"]
device_num_list = [4, 4]
global_batch_size = 32
num_micro_batches = 8
num_layers = 4
```
# Run a Task
Start the auto-tuning: 
```
export PYTHONPATH=/****/FlagScale:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/***/FlagScale/third_party/Megatron-LM

python flagscale/runner/auto_tuner/simulator/config_gen.py
