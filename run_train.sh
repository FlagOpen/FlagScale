export NCCL_DEBUG=DEBUG
# export NCCL_DEBUG_FILE="/share/project/heyongzhe/logs/nccl_debug_file_trace_hetero"
# export NCCL_TOPO_DUMP_FILE="/share/project/heyongzhe/logs/nccl_topo_dump_file_trace_hetero"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_SOCKET_IFNAME=lo
python run.py --config-path ./examples/aquila/conf --config-name config_hetero
