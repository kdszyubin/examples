#!/bin/bash

# Enable nullglob option
shopt -s nullglob

# Directory containing input images
input_dir="/data/dataset/input/fast_ns"
# Directory where the output images will be saved
output_dir="/data/dataset/output/fast_ns"

# Models
declare -a models=("mosaic.pth" "candy.pth" "rain_princess.pth" "udnie.pth" "the_scream.pth")

# Loop over all images in the input directory
for file in $input_dir/*; do
  # Extract the base name of the file (i.e., without the directory part)
  echo $file
  base_name=$(basename "$file")
  # Loop over all models
  for model in "${models[@]}"; do
    # Define the output file name
    output_file="${output_dir}/${model}_${base_name}"
    # Run the style transfer
    python neural_style/neural_style.py eval --content-image "$file" --model "saved_models/${model}" --output-image "$output_file" --cuda 0
  done
done

# Disable nullglob option
shopt -u nullglob

