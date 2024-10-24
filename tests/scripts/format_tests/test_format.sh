#!/bin/bash


# Source the search_py_files.sh script to use the search_py_files function
source "$(dirname "$0")/search_py_files.sh"


# Define the INCLUDE folders to search
INCLUDE_FOLDS=("flagscale")
# Call the function for include folders
INCLUDE_FILES=$(search_py_files "${INCLUDE_FOLDS[@]}")
echo "******************************************** Included files ********************************************"
echo "$INCLUDE_FILES"


# Define the EXCLUDE folders to search
EXCLUDE_FOLDS=("megatron/megatron/core" "megatron/megatron/inference")
# Call the function for exclude folders
EXCLUDE_FILES=$(search_py_files "${EXCLUDE_FOLDS[@]}")
echo "******************************************** Excluded files ********************************************"
echo "$EXCLUDE_FILES"


# Function to run a command and continue even if it fails
run_command() {
  $1
}

echo "******************************************** Running black ********************************************"

include_files=""
for file in $INCLUDE_FILES; do
    include_files+="$file|"
done
echo "$include_files"

exclude_files=""
for file in $EXCLUDE_FILES; do
    exclude_files+="$file|"
done
echo "$exclude_files"

# Now output the changes that were made using black --diff
echo "Showing changes made by black..."
run_command "black --include '$include_files' ./ --exclude '$exclude_files' --diff"

# Run black to format the files
echo "Applying black formatting..."
run_command "black --include '$include_files' ./ --exclude '$exclude_files'"

echo "******************************************** Running isort ********************************************"
echo $INCLUDE_FILES
skip_files=""
for file in $EXCLUDE_FILES; do
    skip_files+="--skip $file "
done
echo $skip_files

echo "Showing changes made by isort..."
run_command "isort --profile black $INCLUDE_FILES $skip_files --diff"

echo "Applying isort formatting..."
run_command "isort --profile black $INCLUDE_FILES $skip_files"
