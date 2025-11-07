# inference_pipeline.py (Self-Contained Configuration)

import argparse
import os
import os.path as osp
import random
import sys

from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import torch

from PIL import Image
from tqdm import tqdm

target_folder = os.path.abspath(
    "/nfs/lhh/luoyc/Emu/Emu3.5"
)  # Local Emu3_5 project path https://github.com/baaivision/Emu3.5
if target_folder not in sys.path:
    sys.path.append(target_folder)

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from src.utils.generation_utils import generate, multimodal_decode
    from src.utils.input_utils import build_image, smart_resize
    from src.utils.logging_utils import setup_logger
    from src.utils.model_utils import build_emu3p5
    from src.utils.painting_utils import ProtoWriter
except ImportError as e:
    print(f"[ERROR] Could not import necessary utility modules: {e}")
    print(
        "[NOTE] Please ensure 'src' directory is in your PYTHONPATH or run from the project root."
    )


# --- Dummy Config Class to hold parameters ---
class DummyConfig:
    """A simple object to hold all configuration parameters, replacing the module import."""

    pass


class InferencePipeline:
    """
    A class to encapsulate the inference logic, initialized with a configuration object.
    """

    def __init__(self):
        """
        Initializes the pipeline, loads the model, tokenizer, and VQ model.
        Accepts command line arguments (args) to initialize the internal config (self.cfg).
        """
        # 1. Initial configuration
        self._initialize_config()

        # 2. Other configuration
        self._setup_config()
        self._load_models()
        self.proto_writer = ProtoWriter()

    def _initialize_config(self):
        """Initializes and configures the DummyConfig object from default values and args."""
        cfg = DummyConfig()

        # 1. Path and model configuration
        cfg.model_path = (
            "/nfs/lhh/luoyc/models/Emu3.5"  # https://www.modelscope.cn/models/BAAI/Emu3.5/files
        )
        cfg.vq_path = "/nfs/lhh/luoyc/models/Emu3.5-VisionTokenizer"  # https://www.modelscope.cn/models/BAAI/Emu3.5-VisionTokenizer/files
        cfg.tokenizer_path = "/nfs/lhh/luoyc/Emu/Emu3.5/src/tokenizer_emu3_ibq"
        cfg.vq_type = "ibq"

        cfg.task_type = "story"
        cfg.use_image = True

        # 2. Save configuration and logs
        cfg.exp_name = "emu3p5"
        cfg.save_path = os.path.join(current_dir, f"outputs/{cfg.exp_name}")
        cfg.save_to_proto = True
        setup_logger(cfg.save_path)  # Call setup_logger utility here

        # 3. Equipment and generated parameters
        cfg.hf_device = "auto"
        cfg.vq_device = "cuda:0"
        cfg.streaming = False
        cfg.unconditional_type = "no_text"
        cfg.classifier_free_guidance = 3.0
        cfg.max_new_tokens = 32768
        cfg.image_area = 518400

        # Get prompt template
        cfg.unc_prompt, cfg.template = build_unc_and_template(cfg.task_type, cfg.use_image)

        # 4. Sampling Parameters
        sampling_params = dict(
            use_cache=True,
            text_top_k=1024,
            text_top_p=0.9,
            text_temperature=1.0,
            image_top_k=10240,
            image_top_p=1.0,
            image_temperature=1.0,
            top_k=131072,
            top_p=1.0,
            temperature=1.0,
            num_beams_per_group=1,
            num_beam_groups=1,
            diversity_penalty=0.0,
            max_new_tokens=cfg.max_new_tokens,
            guidance_scale=1.0,
            use_differential_sampling=True,
        )
        sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
        sampling_params["num_beams"] = (
            sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
        )

        cfg.sampling_params = sampling_params

        # 5. Special Tokens
        cfg.special_tokens = dict(
            BOS="<|extra_203|>",
            EOS="<|extra_204|>",
            PAD="<|endoftext|>",
            EOL="<|extra_200|>",
            EOF="<|extra_201|>",
            TMS="<|extra_202|>",
            IMG="<|image token|>",
            BOI="<|image start|>",
            EOI="<|image end|>",
            BSS="<|extra_100|>",
            ESS="<|extra_101|>",
            BOG="<|extra_60|>",
            EOG="<|extra_61|>",
            BOC="<|extra_50|>",
            EOC="<|extra_51|>",
        )

        cfg.seed = 6666

        # 6. Set worker/rank configuration from command line arguments
        # cfg.rank = args.worker_id
        # cfg.world_size = args.num_workers
        cfg.rank = 0
        cfg.world_size = 1

        self.cfg = cfg

    def _setup_config(self):
        """Sets up configuration parameters and initializes save path."""
        cfg = self.cfg

        # 1. Ensure save path directory structure exists
        os.makedirs(cfg.save_path, exist_ok=True)

        # 2. Worker/rank setup (only needed for setting random seed)
        rank = cfg.rank
        random.seed(cfg.seed + rank)

    def _load_models(self):
        """Loads the main model, tokenizer, and VQ model using paths from cfg."""
        cfg = self.cfg
        hf_device, vq_device = cfg.hf_device, cfg.vq_device

        # The model loading function signature is complex, matching the original script
        model, tokenizer, vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=cfg.vq_type,
            model_device=hf_device,
            vq_device=vq_device,
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )
        print(f"[INFO] Model loaded successfully")

        # Store models and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.vq_model = vq_model

        # Get special token IDs
        cfg.special_token_ids = {}
        for k, v in cfg.special_tokens.items():
            encoded_tokens = self.tokenizer.encode(v)
            cfg.special_token_ids[k] = encoded_tokens[0] if encoded_tokens else None

    @torch.no_grad()
    def forward(
        self, prompt: str, reference_image: Optional[List[str]], name: str = "output_000"
    ) -> str:
        """
        Performs inference for a single given prompt and reference image path(s).

        Args:
            prompt (str): The text prompt for generation.
            reference_image (Optional[List[str]]): A list containing reference image paths, or None (no reference image)
            name (str): A name for the current run, used for saving the proto file.

        Returns:
            str: A message indicating the result (e.g., success message or error).
        """
        cfg = self.cfg
        model = self.model
        tokenizer = self.tokenizer
        vq_model = self.vq_model
        save_path = cfg.save_path

        # Check if the result already exists
        proto_file_path = f"{save_path}/proto/{name}.pb"
        if osp.exists(proto_file_path):
            return f"[INFO] Skipping prompt '{name}'. Result already exists at {proto_file_path}"

        torch.cuda.empty_cache()

        # Unified processing of reference image data
        loaded_ref_image = None

        if reference_image is not None and len(reference_image) > 0:
            print(f"[INFO] {len(reference_image)} reference images are provided for '{name}'")
            loaded_ref_image = [Image.open(img_path).convert("RGB") for img_path in reference_image]
        else:
            print(f"[INFO] No reference image is provided for '{name}'")

        # --- Proto Writer Setup ---
        self.proto_writer.clear()
        self.proto_writer.extend([["question", prompt]])

        if loaded_ref_image is not None:
            for img in loaded_ref_image:
                self.proto_writer.extend([["reference_image", img]])

        # --- Prompt Construction ---
        success = True
        formatted_prompt = cfg.template.format(question=prompt)
        unc_prompt = cfg.unc_prompt

        print(f"[INFO] Handling prompt: {formatted_prompt}")

        force_same_image_size = True
        if loaded_ref_image is not None:
            image_str = ""
            for img in loaded_ref_image:
                image_str += build_image(img, cfg, tokenizer, vq_model)

            if len(loaded_ref_image) > 1:
                force_same_image_size = False

            # Replace the <|IMAGE|> placeholder with image tokens
            formatted_prompt = formatted_prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            # If no image, remove the <|IMAGE|> placeholder if it exists in templates
            formatted_prompt = formatted_prompt.replace("<|IMAGE|>", "")
            unc_prompt = unc_prompt.replace("<|IMAGE|>", "")

        # --- Tokenization and BOS Token prepending ---
        input_ids = tokenizer.encode(
            formatted_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor(
                [[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(
            unc_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        full_unc_ids = None
        if hasattr(cfg, "img_unc_prompt"):
            full_unc_ids = tokenizer.encode(
                cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False
            ).to(model.device)

        # --- Generation Loop ---
        result_message = f"[INFO] Inference finished for '{name}'."
        for result_tokens in generate(
            cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids, force_same_image_size
        ):
            try:
                result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, tokenizer, vq_model)
                self.proto_writer.extend(mm_out)
            except Exception as e:
                success = False
                result_message = f"[ERROR] Failed to generate token sequence for '{name}': {e}"
                print(result_message)
                break

        if not success:
            return result_message

        # Save the result
        if cfg.save_to_proto:
            os.makedirs(f"{save_path}/proto", exist_ok=True)
            self.proto_writer.save(proto_file_path)
            result_message = f"[SUCCESS] Result saved to {proto_file_path}"

        return result_message


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    args = parser.parse_args()
    return args


def build_unc_and_template(task: str, with_image: bool):
    """Replicated helper function from config.py."""
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = (
            "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>"
            % task_str
        )
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = (
            "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>"
            % task_str
        )
    return unc_p, tmpl


def main():
    """Main function to run the pipeline."""
    # 1. Analyze command-line parameters
    # args = parse_args()

    pipeline = InferencePipeline()
    cfg = pipeline.cfg

    name = "test"
    result = pipeline.forward(
        prompt="Tell a story about a clay astronaut exploring Mars and discovering a new continent hidden beneath the red dust.",
        reference_image=["/nfs/lhh/luoyc/Emu/Emu3.5/assets/ref_img.png"],
        name=name,
    )
    print(result)

    # 3. Prompts Configuration - Define the list of prompts that need to be run
    # _prompts_base = [
    #    {
    #        "prompt": "Tell a story about a clay astronaut exploring Mars and discovering a new continent hidden beneath the red dust.",
    #        "reference_image": ["assets/ref_img.png"],
    #    },
    #    {
    #        "prompt": "Describe the scenery of a futuristic city at sunset.",
    #        "reference_image": None,
    #    },
    # ]

    # 4. Distributed sharding logic
    # rank, world_size = cfg.rank, cfg.world_size
    # prompts_to_run = _prompts_base
    # assigned_prompts = prompts_to_run[rank::world_size]

    # print(f"[INFO] Worker {rank}/{world_size} assigned {len(assigned_prompts)} prompts.")

    # 5. Loop call forward function
    # for idx, data in tqdm(enumerate(assigned_prompts), total=len(assigned_prompts), desc=f"Worker {rank} inference"):
    #
    #    name = f"rank{rank}_idx{idx:03d}"
    #
    #    result = pipeline.forward(
    #        prompt=data["prompt"],
    #        reference_image=data["reference_image"],
    #        name=name
    #    )
    #    print(result)

    # print(f"[INFO] All assigned prompts processed by worker {rank}.")


if __name__ == "__main__":
    main()
