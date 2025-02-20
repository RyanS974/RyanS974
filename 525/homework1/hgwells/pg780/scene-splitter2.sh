#!/bin/zsh

# Check if input file is provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"
scene_count=1
output_dir="scenes"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Initialize variables
current_scene=""
in_scene=false

# Read the input file line by line
while IFS= read -r line || [[ -n "$line" ]]; do
    # Check if line starts with "SCENE"
    if [[ "$line" =~ ^SCENE ]]; then
        # If we were already in a scene, save the previous scene
        if $in_scene; then
            printf -v padded_count "%02d" $scene_count
echo "$current_scene" > "${output_dir}/scene_${padded_count}.txt"
            ((scene_count++))
            current_scene=""
        fi
        # Start new scene
        in_scene=true
        current_scene="$line"
    elif $in_scene; then
        # Append line to current scene
        current_scene="${current_scene}"$'\n'"${line}"
    fi
done < "$input_file"

# Save the last scene if there is one
if $in_scene; then
    printf -v padded_count "%02d" $scene_count
echo "$current_scene" > "${output_dir}/scene_${padded_count}.txt"
fi

echo "Processing complete. Created $((scene_count)) scene files in the '$output_dir' directory."
