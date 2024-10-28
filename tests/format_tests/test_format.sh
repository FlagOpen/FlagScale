#!/bin/bash

# Exit immediately if a command exits with a non-zero status, but continue for black and isort
set -e

# Define the shared paths to be formatted
INCLUDE_PATHS="flagscale/auto_tuner/*.py \
flagscale/auto_tuner/prune/*.py \
flagscale/auto_tuner/record/*.py \
flagscale/auto_tuner/search/*.py \
flagscale/runner/*.py \
flagscale/logger.py \
flagscale/patches_utils.py \
flagscale/datasets/sft_dataset.py"

# Function to run a command and continue even if it fails
run_command() {
  $1
}

echo "******************************************** Running black ********************************************"

# Now output the changes that were made using black --diff
echo "Showing changes made by black..."
run_command "black --verbose --include $INCLUDE_PATHS ./ --diff"

# Run black to format the files
echo "Applying black formatting..."
run_command "black --verbose --include $INCLUDE_PATHS ./"

echo "******************************************** Running isort ********************************************"

# Show the changes made by isort
echo "Showing changes made by isort..."
run_command "isort --verbose --profile black $INCLUDE_PATHS --diff"

# Run isort to sort imports and apply changes
echo "Applying isort formatting..."
run_command "isort --verbose --profile black $INCLUDE_PATHS"
