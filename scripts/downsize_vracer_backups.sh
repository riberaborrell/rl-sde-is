#!/bin/bash

data_dir_path=$1

# Find all JSON files starting with "gen" in the current directory and subdirectories
find $data_dir_path -type f -name "gen*.json" | while read -r input_file; do

  # Define the output file name
  output_file="$(dirname "$input_file")/$(basename "$input_file" | sed 's/^gen/model/')"

  # Process the JSON file and keep only the specified entry
  jq '{
    "State Vector Size": ."Problem"."State Vector Size",
    "Action Vector Size": ."Problem"."Action Vector Size",
    "Policy Hyperparameters": ."Solver"."Training"."Current Policies"."Policy Hyperparameters",
    "Neural Network": ."Solver"."Neural Network",
  }' "$input_file" > "$output_file"

  # Check if jq succeeded
  if [ $? -eq 0 ]; then
    echo "Successfully downsized '$input_file'."

    # Delete the original file
    #rm "$input_file"
    echo "Deleted original file."
  else
    echo "Error processing JSON file."
    exit 1
  fi
done
