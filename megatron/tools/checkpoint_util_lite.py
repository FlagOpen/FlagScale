"""
Convert a checkpoint from Megatron-LM based on different parallel configurations.
This code is based on the checkpoint_util.py in Megatron-LM, but is modifiyed to
support the distributed optimizer conversion and not build the model during the conversion.
"""
import argparse
import importlib
import torch.multiprocessing as mp
import os
import sys
import torch
import itertools


def load_plugin(plugin_type, name):
    module_name = f"checkpoint_{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, "add_arguments"):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


def inspect_dict(input, level=0):
    for key in input:
        if isinstance(input[key], dict):
            print(
                "  " * level,
                "Key {} {} of type {}".format(key, type(key), type(input[key])),
                flush=True,
            )
            inspect_dict(input[key], level + 1)
        elif isinstance(input[key], torch.Tensor):
            print(
                "  " * level,
                "Key {} {} of shape {}".format(key, type(key), input[key].shape),
                flush=True,
            )
        else:
            print(
                "  " * level,
                "Key {} {} of type {}".format(key, type(key), type(input[key])),
                flush=True,
            )


def get_model_ckpt_path(ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration):
    directory = "iter_{:07d}".format(iteration)
    pp = False
    if pp_size > 1:
        pp = True
    if not pp:
        common_path = os.path.join(ckpt_dir, directory, f"mp_rank_{tp_rank:02d}")
    else:
        common_path = os.path.join(
            ckpt_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
    return os.path.join(common_path, "model_optim_rng.pt")


def get_model_ckpt_paths(ckpt_dir, tp_size, pp_size, iteration):
    model_ckpt_paths = [
        [
            get_model_ckpt_path(ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration)
            for pp_rank in range(pp_size)
        ]
        for tp_rank in range(tp_size)
    ]
    return model_ckpt_paths


def get_optimizer_ckpt_path(ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration):
    directory = "iter_{:07d}".format(iteration)
    pp = False
    if pp_size > 1:
        pp = True
    if not pp:
        common_path = os.path.join(ckpt_dir, directory, f"mp_rank_{tp_rank:02d}")
    else:
        common_path = os.path.join(
            ckpt_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
    return os.path.join(common_path, "distrib_optim.pt")


def get_optimizer_ckpt_paths(ckpt_dir, tp_size, pp_size, iteration):
    model_ckpt_paths = [
        [
            get_optimizer_ckpt_path(
                ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration
            )
            for pp_rank in range(pp_size)
        ]
        for tp_rank in range(tp_size)
    ]
    return model_ckpt_paths


def get_param_index_map_path(ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration):
    directory = "iter_{:07d}".format(iteration)
    pp = False
    if pp_size > 1:
        pp = True
    if not pp:
        common_path = os.path.join(ckpt_dir, directory, f"mp_rank_{tp_rank:02d}")
    else:
        common_path = os.path.join(
            ckpt_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
    return os.path.join(common_path, "param_name_to_index_maps.json")


def get_param_index_map_paths(ckpt_dir, tp_size, pp_size, iteration):
    param_index_paths = [
        [
            get_param_index_map_path(
                ckpt_dir, tp_size, pp_size, tp_rank, pp_rank, iteration
            )
            for pp_rank in range(pp_size)
        ]
        for tp_rank in range(tp_size)
    ]
    return param_index_paths


def unflatten_optimizer_ckpt(flatten_ckpt, param_name_to_index_map, vp_size, key):
    assert key in ["param", "exp_avg", "exp_avg_sq"]
    assert len(flatten_ckpt) == vp_size
    assert len(param_name_to_index_map) == vp_size
    unflatten_ckpt = {}
    for model_idx in range(vp_size):
        assert len(flatten_ckpt[model_idx]) == 1
        dtype = list(flatten_ckpt[model_idx].keys())[0]
        dtype_flatten_ckpt = list(flatten_ckpt[model_idx].values())[0]
        dtype_param_name_to_index_map = param_name_to_index_map[model_idx]
        model_unflatten_ckpt = {}
        dtype_unflatten_ckpt = {}
        ckpt = dtype_flatten_ckpt[key]
        key_unflatten_ckpt = {}
        for param_name, index_map in dtype_param_name_to_index_map.items():
            shape, index_range = index_map
            param = (
                ckpt[index_range[0] : index_range[1]]
                .clone()
                .reshape(shape)
                .contiguous()
            )
            key_unflatten_ckpt[param_name] = param
        dtype_unflatten_ckpt[key] = key_unflatten_ckpt
        model_unflatten_ckpt[dtype] = dtype_unflatten_ckpt
        unflatten_ckpt[model_idx] = model_unflatten_ckpt
    return unflatten_ckpt


def flatten_optimizer_ckpt(unflatten_ckpt, param_name_to_index_map, vp_size, key):
    assert key in ["param", "exp_avg", "exp_avg_sq"]
    assert len(unflatten_ckpt) == vp_size
    assert len(param_name_to_index_map) == vp_size
    flatten_ckpt = {}
    for model_idx in range(vp_size):
        assert len(unflatten_ckpt[model_idx]) == 1
        dtype = list(unflatten_ckpt[model_idx].keys())[0]
        dtype_unflatten_ckpt = list(unflatten_ckpt[model_idx].values())[0]
        dtype_param_name_to_index_map = param_name_to_index_map[model_idx]
        model_flatten_ckpt = {}
        dtype_flatten_ckpt = {}
        ckpt = dtype_unflatten_ckpt[key]
        num_elements = 0
        for _, param in ckpt.items():
            assert param.dtype == torch.float32
            num_elements += param.nelement()
        key_flatten_ckpt = torch.empty(num_elements, dtype=torch.float32, device="cpu")
        for param_name, index_map in dtype_param_name_to_index_map.items():
            shape, index_range = index_map
            assert shape == list(ckpt[param_name].shape), (
                shape,
                type(shape),
                type(shape[0]),
                list(ckpt[param_name].shape),
                param_name,
            )
            assert index_range[1] - index_range[0] == ckpt[param_name].nelement(), (
                index_range[1] - index_range[0],
                ckpt[param_name].nelement(),
                param_name,
            )
            param = ckpt[param_name].reshape(-1).contiguous()
            key_flatten_ckpt[index_range[0] : index_range[1]].copy_(param)
        dtype_flatten_ckpt[key] = key_flatten_ckpt
        model_flatten_ckpt[dtype] = dtype_flatten_ckpt
        flatten_ckpt[model_idx] = model_flatten_ckpt
    return flatten_ckpt


def split_optimizer_ckpt(ckpt_path, vp_size):
    print(f"Splitting from {ckpt_path} ...")
    ckpt_name, ckpt_ext = os.path.splitext(ckpt_path)
    merged_ckpt = torch.load(ckpt_path, map_location="cpu")
    assert len(merged_ckpt) == vp_size
    for key in ["param", "exp_avg", "exp_avg_sq"]:
        splitted_ckpt = {}
        for model_idx in range(vp_size):
            assert len(merged_ckpt[model_idx]) == 1
            dtype = list(merged_ckpt[model_idx].keys())[0]
            dtype_merged_ckpt = list(merged_ckpt[model_idx].values())[0]
            ckpt = dtype_merged_ckpt[key]
            model_splitted_ckpt = {}
            dtype_splitted_ckpt = {}
            dtype_splitted_ckpt[key] = ckpt
            model_splitted_ckpt[dtype] = dtype_splitted_ckpt
            splitted_ckpt[model_idx] = model_splitted_ckpt
        save_path = ckpt_name + "_" + key + ckpt_ext
        print(f"    {key} is saved to {save_path}.")
        torch.save(splitted_ckpt, save_path)
    print(f"Splitting from {ckpt_path} done.")


def merge_optimizer_ckpt(ckpt_path, vp_size):
    print(f"Merging to {ckpt_path} ...")
    ckpt_name, ckpt_ext = os.path.splitext(ckpt_path)
    merged_ckpt = {}
    for key in ["param", "exp_avg", "exp_avg_sq"]:
        load_path = ckpt_name + "_" + key + ckpt_ext
        print(f"    {key} is loaded from {load_path}.")
        splitted_ckpt = torch.load(load_path, map_location="cpu")
        assert len(splitted_ckpt) == vp_size
        for model_idx in range(vp_size):
            assert len(splitted_ckpt[model_idx]) == 1
            dtype = list(splitted_ckpt[model_idx].keys())[0]
            dtype_splitted_ckpt = list(splitted_ckpt[model_idx].values())[0]
            ckpt = dtype_splitted_ckpt[key]
            if merged_ckpt.get(model_idx, None) is None:
                merged_ckpt[model_idx] = {}
            if merged_ckpt[model_idx].get(dtype, None) is None:
                merged_ckpt[model_idx][dtype] = {}
            merged_ckpt[model_idx][dtype][key] = ckpt
    torch.save(merged_ckpt, ckpt_path)
    print(f"Merging to {ckpt_path} done.")


def remove_optimizer_tmp(ckpt_path):
    print(f"Removing from {ckpt_path} ...")
    ckpt_name, ckpt_ext = os.path.splitext(ckpt_path)
    for key in ["param", "exp_avg", "exp_avg_sq"]:
        path = ckpt_name + "_" + key + ckpt_ext
        if os.path.exists(path):
            os.remove(path)
            print(f"    {path} is removed.")
    print(f"Removing from {ckpt_path} done.")


def get_num_layers_from_ckpt(model_ckpt, vp_size, vp_rank):
    if vp_size > 1:
        keys = model_ckpt["model" + str(vp_rank)]["language_model"]["encoder"].keys()
    else:
        keys = model_ckpt["model"]["language_model"]["encoder"].keys()
    layers = set()
    for key in keys:
        if "layers" in key:
            tmp = key.split(".")[1]
            layers.add(tmp)
    return len(layers)


def get_num_layers_from_args(
    total_num_layers,
    pp_size,
    pp_rank,
    hetero_pipeline_stages=None,
    is_decoder=False,
    standalone_embedding_stage=False,
):
    assert standalone_embedding_stage == False, "standalone_embedding_stage is not supported"
    def _compute_num_layers():
        pipeline_stages = [item for sublist in hetero_pipeline_stages for item in sublist]
        num_layers = pipeline_stages[pp_rank] 
        return num_layers

    if pp_size > 1:
        assert (
            total_num_layers % pp_size == 0
        ), "num_layers must be divisible by transformer_pipeline_model_parallel_size"

        if hetero_pipeline_stages is not None:
            num_layers = _compute_num_layers()
            if standalone_embedding_stage and pp_rank == 0:
                return 0
            else:
                return num_layers
        else:
            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if standalone_embedding_stage and pp_rank == 0
                else total_num_layers // pp_size
            )
    else:
        if not is_decoder:
            num_layers = total_num_layers
        else:
            num_layers = total_num_layers
    return num_layers


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Megatron Checkpoint Utility Arguments",
        allow_abbrev=False,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["GPT"],
        help="Type of the model",
    )
    parser.add_argument(
        "--conversion-type",
        type=str,
        required=True,
        choices=["weight", "optimizer"],
        help="Type of how to convert the checkpoint",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        required=True,
        help="Directory to load model checkpoint from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save model checkpoint to",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=30,
        help="Maximum number of tensors in the queue",
    )
    parser.add_argument(
        "--no-checking",
        action="store_false",
        help="Do not perform checking on the name and ordering of weights",
        dest="checking",
    )
    parser.add_argument(
        "--no-del-tmp",
        action="store_true",
        help="Do not delete temporary files after conversion",
    )
    parser.add_argument(
        "--loader-mapping-from-start",
        action="store_true",
        help="Get param index mapping file from iteration 0 for loader",
        default=False,
    )
    parser.add_argument(
        "--saver-mapping-from-start",
        action="store_true",
        help="Get param index mapping file from iteration 0 for saver",
        default=True,
    )
    parser.add_argument(
        '--hetero-pipeline-stages', nargs='*', type=int, default=None,
        help='Incompatible with --num-layers-per-virtual-pipeline-stage.'
             'hetero-pipeline-stages must be in the form:'
             'n0 layers_0_0 layers_0_1 ... n1 nlayers_1_0 nlayers_1_1 ...'
    )

    known_args, _ = parser.parse_known_args()

    if known_args.conversion_type == "weight":
        # ------ convert model weight ------
        loader = load_plugin("loader", "weight_lite")
        saver = load_plugin("saver", "weight_lite")

        loader.add_arguments(parser)
        saver.add_arguments(parser)

        args = parser.parse_args()

        queue = mp.Queue(maxsize=args.max_queue_size)

        print("Begin to convert model weight...")
        saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
        saver_proc.start()

        loader.load_checkpoint(queue, args)

        saver_proc.join()
        print("Finish converting model weight.")
    elif known_args.conversion_type == "optimizer":
        # ------ convert optimizer state ------
        loader = load_plugin("loader", "optimizer_lite")
        saver = load_plugin("saver", "optimizer_lite")

        loader.add_arguments(parser)
        saver.add_arguments(parser)

        args = parser.parse_args()
        queue = mp.Queue(maxsize=args.max_queue_size)

        print("Begin to convert optimizer state...")
        saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
        saver_proc.start()

        loader.load_checkpoint(queue, args)

        saver_proc.join()
        print("Finish converting optimizer state.")
    else:
        raise ValueError("Unknown type: {}".format(type))


if __name__ == "__main__":
    main()
