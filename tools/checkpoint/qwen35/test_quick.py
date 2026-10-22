#!/usr/bin/env python3
"""Quick test: Qwen 3.6-35B-A3B with expert_tp != tp."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = os.environ.get("QWEN35_MODEL_DIR", "/path/to/models/Qwen3.6-35B-A3B")
TMP_BASE = os.environ.get("QWEN35_TMP_BASE", "/path/to/models/tmp")
CONVERT_SCRIPT = str(Path(__file__).resolve().parent / "convert_qwen35.py")
PYTHON = os.environ.get("QWEN35_PYTHON", sys.executable)

# Single critical test: TP=2, Expert-TP=1
TP, PP, EP, EXPERT_TP = 2, 1, 1, 1

def load_hf_config():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)

def create_yaml(yaml_path, hf_cfg):
    import yaml
    config = {
        "tensor_model_parallel_size": TP,
        "pipeline_model_parallel_size": PP,
        "expert_model_parallel_size": EP,
        "expert_tensor_parallel_size": EXPERT_TP,
        "num_layers": hf_cfg["num_hidden_layers"],
        "hidden_size": hf_cfg["hidden_size"],
        "num_attention_heads": hf_cfg["num_attention_heads"],
        "num_query_groups": hf_cfg["num_key_value_heads"],
        "kv_channels": hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"],
        "attention_output_gate": hf_cfg.get("attention_output_gate", False),
        "untie_embeddings_and_output_weights": not hf_cfg.get("tie_word_embeddings", False),
        "linear_attention_freq": hf_cfg.get("linear_attention_freq", 999),
        "linear_key_head_dim": hf_cfg.get("linear_key_head_dim", 64),
        "linear_value_head_dim": hf_cfg.get("linear_value_head_dim", 64),
        "linear_num_key_heads": hf_cfg.get("linear_num_key_heads", 16),
        "linear_num_value_heads": hf_cfg.get("linear_num_value_heads", 16),
        "num_experts": hf_cfg.get("num_experts", 0),
        "moe_ffn_hidden_size": hf_cfg.get("moe_intermediate_size", 0),
        "moe_shared_expert_intermediate_size": hf_cfg.get("shared_expert_intermediate_size", 0),
        "ffn_hidden_size": hf_cfg.get("intermediate_size", hf_cfg.get("moe_intermediate_size", 0)),
        "no_enable_vision": True,
        "mtp_num_layers": 0,
    }
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

def run_cmd(cmd, label):
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
            print(f"  stderr tail:\n{result.stderr[-1000:]}")
            return False
        print(f"  SUCCESS ({elapsed:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 3600s")
        return False

def main():
    print(f"Testing TP={TP}, PP={PP}, EP={EP}, Expert-TP={EXPERT_TP}")
    print(f"Model: {MODEL_DIR}")

    test_dir = os.path.join(TMP_BASE, "quick_test")
    os.makedirs(test_dir, exist_ok=True)

    yaml_path = os.path.join(test_dir, "train.yaml")
    meg_dir = os.path.join(test_dir, "megatron")
    hf_out = os.path.join(test_dir, "hf_recovered")

    try:
        print("\n[1/3] Creating config...")
        hf_cfg = load_hf_config()
        create_yaml(yaml_path, hf_cfg)
        print(f"  Model layers: {hf_cfg['num_layers']}, experts: {hf_cfg.get('num_experts', 0)}")

        print("\n[2/3] HF -> Megatron...")
        cmd = [PYTHON, CONVERT_SCRIPT, "--direction", "hf2meg",
               "--hf-path", MODEL_DIR, "--meg-path", meg_dir, "--yaml", yaml_path]
        if not run_cmd(cmd, "HF->Megatron"):
            return False

        # Check shards
        release = os.path.join(meg_dir, "release")
        if os.path.exists(release):
            shards = [d for d in os.listdir(release) if d.startswith("mp_rank_")]
            print(f"  Generated {len(shards)} shards")
            for s in shards[:5]:
                print(f"    {s}")

        print("\n[3/3] Megatron -> HF...")
        cmd = [PYTHON, CONVERT_SCRIPT, "--direction", "meg2hf",
               "--meg-path", meg_dir, "--hf-path", hf_out, "--yaml", yaml_path]
        if not run_cmd(cmd, "Megatron->HF"):
            return False

        print("\n✓ Conversion succeeded!")
        print(f"  Output: {hf_out}")
        return True

    finally:
        # Clean up
        if os.path.exists(test_dir):
            print(f"\nCleaning up {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
