import sys
import argparse
import importlib
import torch.multiprocessing as mp


def load_plugin(plugin_type, name):
    module_name = f"{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoint")
    # convert args
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['aquila', 'mixtral'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, shdoul be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')

    known_args, _ = parser.parse_known_args()
    loader = load_plugin('loader', known_args.loader)
    saver = load_plugin('saver', known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)
    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()
