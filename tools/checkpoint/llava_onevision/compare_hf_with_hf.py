import argparse
import os
import torch
from safetensors.torch import load_file

def compare_transformer_models(checkpoint_path_A, checkpoint_path_B):
    # Load model state dictionaries from the specified paths
    model_state_dict_A = load_file(checkpoint_path_A)
    model_state_dict_B = load_file(checkpoint_path_B)

    # Check if both models have the same weights
    missing_keys_A = []
    missing_keys_B = []

    for key in model_state_dict_A:
        if key not in model_state_dict_B:
            missing_keys_B.append(key)

    for key in model_state_dict_B:
        if key not in model_state_dict_A:
            missing_keys_A.append(key)

    if missing_keys_A:
        print("*" * 30)
        print("The following keys are missing in Model A:", missing_keys_A)

    if missing_keys_B:
        print("*" * 30)
        print("The following keys are missing in Model B:", missing_keys_B)

    # Compare the parameters
    for key in model_state_dict_A:
        if key in model_state_dict_B:
            tensor_A = model_state_dict_A[key]
            tensor_B = model_state_dict_B[key]

            # Check for shape mismatch
            if tensor_A.shape != tensor_B.shape:
                print(f"Shape mismatch for key '{key}': {tensor_A.shape} vs {tensor_B.shape}")
            else:
                # Compare tensors
                if torch.equal(tensor_A, tensor_B):
                    print(f"Tensors for key '{key}' are equal.")
                else:
                    print(f"Tensors for key '{key}' are NOT equal.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" 
        Compare transformers' weights from two Hugging Face format checkpoint files. 
        Example usage:
        python compare_hf_with_hf.py --checkpoint_path_A ./path_to_checkpoint_A --checkpoint_path_B ./path_to_checkpoint_B
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--checkpoint_path_A", type=str, required=True, help="Path to the first checkpoint file in Hugging Face format.")
    parser.add_argument("--checkpoint_path_B", type=str, required=True, help="Path to the second checkpoint file in Hugging Face format.")

    args = parser.parse_args()

    compare_transformer_models(args.checkpoint_path_A, args.checkpoint_path_B)

    print("Comparison done.")
