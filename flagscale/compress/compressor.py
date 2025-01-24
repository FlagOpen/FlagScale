import os
import sys
import argparse
import yaml
import shutil
from omegaconf import OmegaConf

import torch
from transformers import *
from llmcompressor.transformers.sparsification.compressed_tensors_utils import modify_save_pretrained
from llmcompressor.utils.pytorch.module import set_layer

from flagscale.compress.combined_algo import prepare_compress_methods
from flagscale.compress.adapter import LLMCompressorAdapter
from flagscale.compress.algo import RTNWrapper

_g_ignore_fields = ["experiment", "action"]

def prepare_config(config_path):
    # Open the YAML file and convert it into a dictionary
    with open(config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # Extract valid config
    for key in _g_ignore_fields:
        yaml_dict.pop(key)
    new_yaml_dict = {}
    for k, v in yaml_dict.items():
        assert isinstance(
            v, dict
        ), f"Expected a dictionary for key {k}, but got {v} instead"
        new_yaml_dict.update(v)
    config = OmegaConf.create(new_yaml_dict)
    return config

def copy_rest_file(src_path, dst_path):
    from huggingface_hub import hf_hub_download
    from transformers import TRANSFORMERS_CACHE
    from transformers.utils import http_user_agent

    if not os.path.exists(src_path):
        user_agent = http_user_agent()
        config_file_path = hf_hub_download(
            repo_id=src_path,
            filename="config.json",
            cache_dir=TRANSFORMERS_CACHE,
            force_download=False,
            user_agent=user_agent,
        )
        src_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

    dst_path_files = os.listdir(dst_path)
    for filename in os.listdir(src_path):
        if not filename.endswith(".safetensors") and filename not in dst_path_files:
            full_file_name = os.path.join(src_path, filename)
            if (not filename.endswith(".md")) and os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst_path)
            elif os.path.isdir(full_file_name):
                shutil.copytree(full_file_name, os.path.join(dst_path, filename))

class Compressor:
    def __init__(self, cfg, model=None, dataset=None):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset

    def compress(self):
        self.tokenizer = None
        self.model_path = self.cfg.model.pop("model_path")
        if self.cfg.data.tokenzier_args is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.data.tokenzier_args.pop("tokenizer_path"), **self.cfg.data.tokenzier_args)
        if self.model is None:
            model_cls = eval(self.cfg.model.pop("model_cls"))
            self.model = model_cls.from_pretrained(self.model_path, **self.cfg.model)
        # import pdb; pdb.set_trace()
        assert isinstance(self.model, torch.nn.Module), f"model type {type(self.model)} error, please check it"
        compress_args = self.cfg.compress_args
        recipes = prepare_compress_methods(compress_args)
        for method, recipe in recipes.items():
            for algo_args in recipe:
                algo_args = OmegaConf.to_container(algo_args)
                algo_args["dataset"] = self.dataset
                print("algo_args: ", algo_args)
                # import pdb; pdb.set_trace()
                algo_args["num_calibration_steps"] = self.cfg.data.get("num_calibration_steps", 384)
                adapter = LLMCompressorAdapter(model=self.model, **algo_args)
                ### modify model inplace
                self.model = adapter.model
                
            # oneshot(model=model, dataset=dataset, recipe=recipe, tokenizer=tokenizer, output_dir=cfg.system.save_dir, max_seq_length=cfg.data.get("max_seq_length", 384), num_calibration_samples=cfg.data.get("num_calibration_samples", 512), splits="calibration")
    def save_pretrained(self, save_compressed=True):
        modify_save_pretrained(self.model)
        self.model.save_pretrained(self.cfg.system.save_dir, save_compressed=save_compressed)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.cfg.system.save_dir)
        copy_rest_file(self.model_path, cfg.system.save_dir)
        import pdb; pdb.set_trace()

    @torch.no_grad()
    def convert(self, model):
        for name, mod in model.named_modules():
            if hasattr(mod, "weight_scale"):
                wrapper = RTNWrapper(name, mod, enable_fake_quant=True)
                set_layer(name, wrapper, model)
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    cfg = prepare_config(args.config_path)

    Compressor(cfg)
