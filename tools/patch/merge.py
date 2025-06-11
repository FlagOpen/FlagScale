import argparse
import yaml
import os
from git.repo import Repo

from patch import normalize_backend

def merge(backends, device_type, tasks, commit):
    """
    Merge patch history for the specified backend(s), device type, and task(s).
    """
    if not commit:
        raise ValueError("Commit ID must be provided.")
    if not device_type:
        raise ValueError("Device type must be provided.")
    patch_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.dirname(patch_dir)
    FlagScale_dir = os.path.dirname(tools_dir)
    main_repo = Repo(FlagScale_dir)
    try:
        main_repo.commit(commit)
    except Exception as e:
        raise ValueError(f"Invalid commit ID: {commit}. Error: {e}")
    history_yaml = os.path.join(FlagScale_dir, "hardware", "patch_history.yaml")
    if os.path.exists(history_yaml):
        with open(history_yaml, "r") as f:
            history = yaml.safe_load(f) or {}
    else:
        history = {}

    if device_type not in history:
        history[device_type] = {}

    backends_key = "+".join(sorted(backends))

    for task in tasks:
        if task not in history[device_type]:
            history[device_type][task] = {}
        if backends_key not in history[device_type][task]:
            history[device_type][task][backends_key] = []
        if commit not in history[device_type][task][backends_key]:
            history[device_type][task][backends_key].append(commit)

        # Also write to each individual backend
        for backend in backends:
            if backend not in history[device_type][task]:
                history[device_type][task][backend] = []
            if commit not in history[device_type][task][backend]:
                history[device_type][task][backend].append(commit)

    with open(history_yaml, "w") as f:
        yaml.dump(history, f, sort_keys=False)
    print(f"Patch history updated in {history_yaml}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add patch history.")
    parser.add_argument(
        "--backend",
        nargs="+",
        type=normalize_backend,
        default=["Megatron-LM"],
        help="Backend to patch (default: Megatron-LM)",
    )
    parser.add_argument(
        "--device-type", type=str, required=True, help="Device type (e.g., gpu, cambricon, ascend)."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        required=True,
        choices=["train", "inference", "post_train"],
        help="Task(s) to add.",
    )
    parser.add_argument(
        "--commit", type=str, required=True, help="Commit ID to record."
    )

    args = parser.parse_args()

    merge(args.backend, args.device_type, args.task, args.commit)
