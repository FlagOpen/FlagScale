#!/bin/bash

# Function to search for .py files in given folders
search_py_files() {
  local folders=("$@")  # Get the list of folders passed as arguments
  local files=""        # Initialize an empty string to store found files

  # Loop through each folder and search for all .py files recursively
  for folder in "${folders[@]}"; do
    if [ -d "$folder" ]; then
      # Recursively search for all .py files in the folder and its subfolders
      py_files=$(find "$folder" -type f -name "*.py")
      
      # If files are found, append them to the result string
      if [ -n "$py_files" ]; then
        files="$files $py_files"
      fi
    else
      # Print a warning if the folder doesn't exist
      echo "Directory $folder does not exist"
    fi
  done

  # Remove any leading whitespace from the resulting string
  files=$(echo "$files" | sed 's/^ *//')
  echo "$files"  # Output the resulting string
}

# Check if the script is being executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Call the function with command-line arguments
  search_py_files "$@"
fi
