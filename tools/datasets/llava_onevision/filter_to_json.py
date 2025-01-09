import os
import re
import json
import argparse
from typing import Dict


def main():
    parser = argparse.ArgumentParser(description='Grep id and loss from log files.')
    parser.add_argument('--input_dir', type=str, help='Directory to search log files.')
    parser.add_argument('--output', type=str, help='Path to save the result.')
    args = parser.parse_args()

    result_dict: Dict[str, float] = {}
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        match = re.search(r'Evaluating id: (\d+), loss: ([\d.]+)', line)
                        if match:
                            evaluating_id = match.group(1)
                            loss = float(match.group(2))
                            if evaluating_id in result_dict:
                                assert loss == result_dict[evaluating_id]
                            # Customize filtering rules such as
                            # if loss < 0.5:
                            #    result_dict[evaluating_id] = loss

                            # NOTE: No filtering currently, Comment out if Customize
                            result_dict[evaluating_id] = loss

    ids = list(result_dict.keys())
    print("Keep id count: ", len(ids))
    result = {"ids": ids}
    assert args.output.endswith(".json")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
    print("Done")


if __name__ == "__main__":
    main()