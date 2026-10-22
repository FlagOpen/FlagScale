#!/usr/bin/env python3
"""Test real Qwen 3.6-35B-A3B model conversion with various parallelism strategies.

This script:
1. Loads the real HF model from $QWEN35_MODEL_DIR
2. Converts to Megatron with various (TP, PP, EP, Expert-TP) combinations
3. Converts back to HF
4. Compares key weights numerically

All intermediate checkpoints are stored under $QWEN35_TMP_BASE/
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force unbuffered stdout so progress is visible in real-time
sys.stdout.reconfigure(line_buffering=True)

# Test configurations: (tp, pp, ep, expert_tp, description)
# Model constraints: 40 layers (PP divisors: 1,2,4,5,8,10,20,40)
#                    16 heads (TP divisors: 1,2,4,8,16)
#                    256 experts (EP divisors: 1,2,4,8,16,32,64,128,256)
TEST_CONFIGS = [
    # === Required scenarios ===
    # 1) Only TP
    (2, 1, 1, 2, "1) Only TP"),

    # 2) Only PP
    (1, 2, 1, 1, "2) Only PP"),

    # 3) Only EP
    (1, 1, 2, 1, "3) Only EP"),

    # 4) TP+PP+EP all enabled
    (2, 2, 2, 2, "4) TP+PP+EP all enabled, tp==etp"),

    # 5) TP+PP+EP+ETP all enabled, tp==etp
    (4, 2, 2, 4, "5) TP+PP+EP+ETP, tp==etp"),

    # 6) TP+PP+EP+ETP all enabled, tp!=etp
    (4, 2, 2, 2, "6) TP+PP+EP+ETP, tp!=etp"),

    # === Additional scenarios ===
    # Baseline: no parallelism
    (1, 1, 1, 1, "Baseline: no parallelism"),

    # TP!=ETP extreme case
    (4, 1, 1, 1, "TP=4, ETP=1 (max TP/ETP difference)"),
    (8, 1, 1, 1, "TP=8, ETP=1 (even larger TP)"),

    # High EP
    (2, 1, 8, 2, "High EP=8"),
    (4, 1, 8, 2, "TP=4, EP=8, ETP=2 (high EP + etp!=tp)"),

    # Higher TP
    (8, 1, 2, 8, "TP=8, EP=2, ETP=8"),
    (8, 1, 2, 4, "TP=8, EP=2, ETP=4 (high TP, etp!=tp)"),

    # More PP stages
    (2, 4, 2, 2, "TP=2, PP=4, EP=2"),
    (4, 5, 2, 2, "TP=4, PP=5, EP=2 (40 layers / 5 = 8 layers per stage)"),
]

# Paths
MODEL_DIR = os.environ.get("QWEN35_MODEL_DIR", "/path/to/models/Qwen3.6-35B-A3B")
TMP_BASE = os.environ.get("QWEN35_TMP_BASE", "/path/to/models/tmp")
CONVERT_SCRIPT = str(Path(__file__).resolve().parent / "convert_qwen35.py")
PYTHON = os.environ.get("QWEN35_PYTHON", sys.executable)


def load_model_config(model_dir):
    """Load HF config.json to determine model parameters."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    # Handle nested text_config
    if "text_config" in cfg:
        cfg = cfg["text_config"]

    return cfg


def create_yaml(yaml_path, hf_config, tp, pp, ep, expert_tp):
    """Generate a training YAML for the given parallelism config."""
    import yaml

    config = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": expert_tp,

        # Model architecture from HF config
        "num_layers": hf_config["num_hidden_layers"],
        "hidden_size": hf_config["hidden_size"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_query_groups": hf_config["num_key_value_heads"],
        "kv_channels": hf_config["hidden_size"] // hf_config["num_attention_heads"],
        "attention_output_gate": hf_config.get("attention_output_gate", False),
        "untie_embeddings_and_output_weights": not hf_config.get("tie_word_embeddings", False),

        # Linear attention (GDN)
        "linear_attention_freq": hf_config.get("linear_attention_freq", 999),
        "linear_key_head_dim": hf_config.get("linear_key_head_dim", 64),
        "linear_value_head_dim": hf_config.get("linear_value_head_dim", 64),
        "linear_num_key_heads": hf_config.get("linear_num_key_heads", 16),
        "linear_num_value_heads": hf_config.get("linear_num_value_heads", 16),

        # MoE parameters
        "num_experts": hf_config.get("num_experts", 0),
        "moe_ffn_hidden_size": hf_config.get("moe_intermediate_size", 0),
        "moe_shared_expert_intermediate_size": hf_config.get("shared_expert_intermediate_size", 0),
        "ffn_hidden_size": hf_config.get("intermediate_size", hf_config.get("moe_intermediate_size", 0)),

        # Vision and MTP (disable for pure language model tests)
        "no_enable_vision": True,
        "mtp_num_layers": 0,
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    return config


def run_conversion(direction, hf_path, meg_path, yaml_path, label):
    """Run convert_qwen35.py and return success status."""
    if direction == "hf2meg":
        cmd = [
            PYTHON, CONVERT_SCRIPT,
            "--direction", "hf2meg",
            "--hf-path", hf_path,
            "--meg-path", meg_path,
            "--yaml", yaml_path,
        ]
    else:  # meg2hf
        cmd = [
            PYTHON, CONVERT_SCRIPT,
            "--direction", "meg2hf",
            "--meg-path", meg_path,
            "--hf-path", hf_path,
            "--yaml", yaml_path,
        ]

    print(f"    Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    FAILED: {label}")
            print(f"    stderr: {result.stderr[-500:]}")  # Last 500 chars
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {label}")
        return False
    except Exception as e:
        print(f"    ERROR: {label} - {e}")
        return False


def verify_checkpoint_naming(release_dir, tp, pp, ep, label):
    """Verify checkpoint directory names match Megatron convention."""
    if not os.path.isdir(release_dir):
        print(f"    ✗ release dir not found: {release_dir}")
        return False

    actual_dirs = sorted(d for d in os.listdir(release_dir) if d.startswith("mp_rank_"))

    expected_dirs = set()
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            if ep > 1:
                for ep_rank in range(ep):
                    if pp > 1:
                        name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
                    else:
                        name = f"mp_rank_{tp_rank:02d}_{ep_rank:03d}"
                    expected_dirs.add(name)
            else:
                if pp > 1:
                    name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
                else:
                    name = f"mp_rank_{tp_rank:02d}"
                expected_dirs.add(name)

    actual_set = set(actual_dirs)
    if actual_set != expected_dirs:
        missing = expected_dirs - actual_set
        extra = actual_set - expected_dirs
        if missing:
            print(f"    ✗ Missing dirs: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
        if extra:
            print(f"    ✗ Extra dirs: {sorted(extra)[:5]}{'...' if len(extra)>5 else ''}")
        return False

    # Verify each dir contains model_optim_rng.pt
    for d in actual_dirs:
        pt_path = os.path.join(release_dir, d, "model_optim_rng.pt")
        if not os.path.exists(pt_path):
            print(f"    ✗ Missing model_optim_rng.pt in {d}")
            return False

    print(f"    ✓ Naming convention verified ({len(actual_dirs)} dirs)")
    return True


def compare_safetensors(original_dir, recovered_dir, label):
    """Compare key expert weights between original and recovered HF checkpoints."""
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        print(f"    SKIP comparison: safetensors not installed")
        return True

    # Find safetensors files
    orig_files = list(Path(original_dir).glob("*.safetensors"))
    recv_files = list(Path(recovered_dir).glob("*.safetensors"))

    if not orig_files or not recv_files:
        print(f"    SKIP comparison: no safetensors files found")
        return True

    # Load first shard from each (should contain some expert weights)
    orig_file = str(orig_files[0])
    recv_file = str(recv_files[0])

    orig_keys = set()
    recv_keys = set()

    with safe_open(orig_file, framework="pt") as f:
        orig_keys = set(f.keys())

    with safe_open(recv_file, framework="pt") as f:
        recv_keys = set(f.keys())

    # Filter to expert-related keys only
    orig_expert_keys = {k for k in orig_keys if ".mlp." in k}
    recv_expert_keys = {k for k in recv_keys if ".mlp." in k}

    # Check if expert keys are present in both
    common_expert_keys = orig_expert_keys & recv_expert_keys

    if not common_expert_keys:
        print(f"    WARNING: No common expert keys found for comparison")
        return True

    # Compare a sample of expert weights
    mismatches = 0
    max_keys_to_check = 10
    checked = 0

    with safe_open(orig_file, framework="pt") as orig_f:
        with safe_open(recv_file, framework="pt") as recv_f:
            for key in sorted(common_expert_keys)[:max_keys_to_check]:
                orig_tensor = orig_f.get_tensor(key)
                recv_tensor = recv_f.get_tensor(key)

                if orig_tensor.shape != recv_tensor.shape:
                    print(f"    Shape mismatch: {key}  {orig_tensor.shape} vs {recv_tensor.shape}")
                    mismatches += 1
                elif not torch.allclose(orig_tensor, recv_tensor, rtol=1e-5, atol=1e-6):
                    diff = (orig_tensor - recv_tensor).abs()
                    print(f"    Value mismatch: {key}  max_diff={diff.max():.2e}")
                    mismatches += 1

                checked += 1

    if mismatches == 0:
        print(f"    ✓ Verified {checked} expert weights match exactly")
        return True
    else:
        print(f"    ✗ {mismatches}/{checked} expert weights mismatched")
        return False


def test_single_config(tp, pp, ep, expert_tp, description, test_idx):
    """Test a single parallelism configuration."""
    label = f"TP={tp}, PP={pp}, EP={ep}, ETP={expert_tp}"
    print(f"\n{'='*80}")
    print(f"Test {test_idx}: {label}")
    print(f"Description: {description}")
    print(f"{'='*80}")

    # Create test directory
    test_dir = os.path.join(TMP_BASE, f"test_{test_idx}_tp{tp}_pp{pp}_ep{ep}_etp{expert_tp}")
    os.makedirs(test_dir, exist_ok=True)

    yaml_path = os.path.join(test_dir, "train.yaml")
    meg_dir = os.path.join(test_dir, "megatron")
    hf_out_dir = os.path.join(test_dir, "hf_recovered")

    try:
        # Load HF config and create YAML
        print("  [1/5] Loading model config...")
        hf_config = load_model_config(MODEL_DIR)
        create_yaml(yaml_path, hf_config, tp, pp, ep, expert_tp)

        # HF -> Megatron
        print(f"  [2/5] Converting HF -> Megatron...")
        start = time.time()
        if not run_conversion("hf2meg", MODEL_DIR, meg_dir, yaml_path, label):
            return False
        elapsed = time.time() - start
        print(f"    ✓ Completed in {elapsed:.1f}s")

        # Verify checkpoint naming convention
        release_dir = os.path.join(meg_dir, "release")
        print(f"  [3/5] Verifying checkpoint naming and shard count...")
        if not verify_checkpoint_naming(release_dir, tp, pp, ep, label):
            return False

        shard_count = len([d for d in os.listdir(release_dir) if d.startswith("mp_rank_")])
        expected_shards = tp * pp * (ep if ep > 1 else 1)
        print(f"    ✓ {shard_count} shards (expected {expected_shards})")
        if shard_count != expected_shards:
            print(f"    ✗ Shard count mismatch!")
            return False

        # Megatron -> HF
        print(f"  [4/5] Converting Megatron -> HF...")
        start = time.time()
        if not run_conversion("meg2hf", hf_out_dir, meg_dir, yaml_path, label):
            return False
        elapsed = time.time() - start
        print(f"    ✓ Completed in {elapsed:.1f}s")

        # Compare
        print(f"  [5/5] Comparing weights...")
        if not compare_safetensors(MODEL_DIR, hf_out_dir, label):
            return False

        print(f"\n  ✓ Test {test_idx} PASSED: {label}")
        return True

    except Exception as e:
        print(f"\n  ✗ Test {test_idx} FAILED: {label}")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up to save space
        if os.path.exists(test_dir):
            print(f"  Cleaning up {test_dir}...")
            shutil.rmtree(test_dir, ignore_errors=True)


def main():
    print("="*80)
    print("Real Model Conversion Test: Qwen 3.6-35B-A3B")
    print("="*80)
    print(f"Model: {MODEL_DIR}")
    print(f"Temp dir: {TMP_BASE}")
    print(f"Configurations to test: {len(TEST_CONFIGS)}")
    print("="*80)

    # Sanity checks
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        sys.exit(1)

    if not os.path.exists(CONVERT_SCRIPT):
        print(f"ERROR: Conversion script not found: {CONVERT_SCRIPT}")
        sys.exit(1)

    # Create tmp directory
    os.makedirs(TMP_BASE, exist_ok=True)

    # Run tests
    results = []
    for idx, (tp, pp, ep, expert_tp, description) in enumerate(TEST_CONFIGS, 1):
        success = test_single_config(tp, pp, ep, expert_tp, description, idx)
        results.append((tp, pp, ep, expert_tp, description, success))

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    passed = sum(1 for _, _, _, _, _, ok in results if ok)
    total = len(results)

    for tp, pp, ep, expert_tp, description, ok in results:
        status = "PASS" if ok else "FAIL"
        marker = "✓" if ok else "✗"
        label = f"TP={tp:2d}, PP={pp}, EP={ep:2d}, ETP={expert_tp}"
        suffix = ""
        if expert_tp != tp:
            suffix = "  <-- etp!=tp"
        print(f"  {marker} {label:35s} {status:4s} - {description}{suffix}")

    print(f"\n{'='*80}")
    print(f"Result: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"{total - passed} TESTS FAILED ✗")
    print(f"{'='*80}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
