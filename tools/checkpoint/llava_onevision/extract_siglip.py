# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

from safetensors.torch import load_file, save_file


def convert(input_path, output_path):
    device = "cuda"

    state_dict_llava = load_file(os.path.join(input_path, "model.safetensors"), device=device)
    state_dict_siglip = load_file(os.path.join(output_path, "model.safetensors"), device=device)
    
    for name, tensor in state_dict_llava.items():        
        if "model.vision_tower.vision_tower.vision_model" in name:
            name_siglip = name.replace("model.vision_tower.vision_tower.", "")
            if name_siglip in state_dict_siglip and state_dict_siglip[name_siglip].shape == tensor.shape:
                state_dict_siglip[name_siglip] = tensor
                print(name, "to" , name_siglip)
            else:
                exit("Output error: Tensor shape mismatch")
    
    # Save the updated state_dict_siglip to the output path
    output_file_path = os.path.join(output_path, "model.safetensors")
    save_file(state_dict_siglip, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Extract SigLIP VIT weights from Llava One Vision.

Example usage:
python extract_siglip.py --input ${Folder containing llava ckpt} --output ${Folder containing siglip ckpt}
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Llava weights folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output directory for SigLIP state dict file(s)",
    )

    args = parser.parse_args()

    convert(args.input, args.output)

    print("done.")
