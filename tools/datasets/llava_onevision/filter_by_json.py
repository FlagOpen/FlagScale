
import os
import json
import argparse
import webdataset as wds
import tarfile
from multiprocessing import Pool


def find_tar_files(input_dir):
    tar_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tar'):
                input_tarfile = os.path.join(root, file)
                tar_files.append(input_tarfile)
    assert tar_files
    return tar_files
                

def process_tar_files_in_node(tar_files, input_dir, output_dir, ids_to_keep):
    num_tar_files_in_node = len(tar_files)
    assert num_tar_files_in_node > 0
    if num_tar_files_in_node == 1:
        num_process = 1
    else:
        num_process = num_tar_files_in_node // 2
        num_process = 8 if num_process >= 8 else num_process

    num_tar_files_per_process = num_tar_files_in_node // num_process
    for i in range(num_process):
        start = i * num_tar_files_per_process
        end = (i + 1) * num_tar_files_per_process if i!= (num_process-1) else num_tar_files_in_node
        process_files = tar_files[start:end]
        with Pool(num_process) as p:
            p.starmap(process_tarfile, [(tf, input_dir, output_dir, ids_to_keep) for tf in process_files])


def process_tarfile(input_tarfile, input_dir, output_dir, ids_to_keep):
    print(f"Process {os.getpid()} is processing {input_tarfile}")
    relative_path = os.path.relpath(input_tarfile, input_dir)
    output_tarfile = os.path.join(output_dir, relative_path)
    output_tarfile_dir = os.path.dirname(output_tarfile)
    os.makedirs(output_tarfile_dir, exist_ok=True)

    dataset = wds.WebDataset(input_tarfile, shardshuffle=False)
    keep_samples = []

    for sample in dataset:
        if sample["__key__"] not in ids_to_keep:
            print(f"id {sample['__key__']} is filtered out.")
        else:
            new_sample = {"__key__": sample["__key__"], "sequence.pyd": sample["sequence.pyd"]}
            keep_samples.append(new_sample)
    if not keep_samples:
        print(f"All id in {input_tarfile} are filtered out.")
    else:
        output_dataset = wds.TarWriter(output_tarfile)
        print(f"Writing {input_tarfile} to {output_tarfile} ...")
        for sample in keep_samples:
            output_dataset.write(sample)
        output_dataset.close()
        print(f"Writing {output_tarfile} done.")

def main():
    parser = argparse.ArgumentParser(description='Filter a webdataset and save the filtered samples.')
    parser.add_argument('--input_dir', type=str, help='Original Directory')
    parser.add_argument('--output_dir', type=str, help='Directory to save the filtered data')
    parser.add_argument('--json_file', type=str, help='Path to the JSON file containing ids')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    json_file = args.json_file

    # Node size
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    # Node rank
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

    with open(json_file, 'r') as f:
        ids_to_keep = json.load(f)["ids"]

    tar_files = find_tar_files(input_dir)
    num_tar_files = len(tar_files)
    assert num_tar_files > size
    num_tar_files_per_node = num_tar_files // size
    start_index = rank * num_tar_files_per_node
    end_index = (rank + 1) * num_tar_files_per_node if rank!= size - 1 else num_tar_files
    node_tar_files = tar_files[start_index:end_index]
    print(f"node_tar_files: ", node_tar_files)
    process_tar_files_in_node(node_tar_files, input_dir, output_dir, ids_to_keep)
    print("Done")



if __name__ == "__main__":
    main()
