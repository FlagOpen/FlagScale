import argparse
import os
import torch
from safetensors.torch import load_file

def load_and_compare(base_dir1, base_dir2):
    rank_files1 = {}
    rank_files2 = {}

    # Traverse the directories for files
    for (root1, _, files1), (root2, _, files2) in zip(os.walk(base_dir1), os.walk(base_dir2)):
        parent_dir1 = os.path.basename(root1)
        parent_dir2 = os.path.basename(root2)
        if parent_dir1.startswith('mp_rank') and parent_dir2.startswith('mp_rank'):
            rank_files1[parent_dir1] = [os.path.join(root1, f) for f in files1 if f.endswith('model_optim_rng.pt')]
            rank_files2[parent_dir2] = [os.path.join(root2, f) for f in files2 if f.endswith('model_optim_rng.pt')]

    for rank in rank_files1.keys():
        print(f"Comparing files in {rank}:")
        files1 = rank_files1[rank]
        files2 = rank_files2[rank]

        for file1, file2 in zip(sorted(files1), sorted(files2)):
            ckpt1 = torch.load(file1)
            ckpt2 = torch.load(file2)
                
            model_1 = ckpt1["model"]
            model_2 = ckpt2["model"]
            
            # Check for missing keys in both models
            for key in model_1:
                if model_1[key] is not None and key not in model_2:
                    print("*" * 30, "Key not found in model_2:", key)
            
            for key in model_2:
                if model_2[key] is not None and key not in model_1:
                    print("*" * 30, "Key not found in model_1:", key)
           
            # Compare tensors in both models
            for key in model_1:
                if model_1[key] is not None and model_2[key] is not None:
                    if model_1[key].shape != model_2[key].shape:
                        print(f"Shape mismatch for key '{key}': {model_1[key].shape} vs {model_2[key].shape}")
                    else:
                        # Move tensors to the same device before comparison
                        tensor1 = model_1[key].to('cuda:0')  # or 'cuda:1'
                        tensor2 = model_2[key].to('cuda:0')  # or 'cuda:1'
                        if torch.equal(tensor1, tensor2):
                            print(f"Tensors for key '{key}' are equal.")
                        else:
                            print(f"Tensors for key '{key}' are NOT equal.")
                            exit(f"Tensors for key '{key}' are NOT equal.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" 
        Compare model weights from two checkpoint directories. 
        Example usage:
        python compare_fs_with_fs.py --input_dir_A ./path_to_directory_A --input_dir_B ./path_to_directory_B
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input_dir_A", type=str, required=True, help="Path to the first checkpoint directory.")
    parser.add_argument("--input_dir_B", type=str, required=True, help="Path to the second checkpoint directory.")

    args = parser.parse_args()

    load_and_compare(args.input_dir_A, args.input_dir_B)

    print("Comparison done.")
