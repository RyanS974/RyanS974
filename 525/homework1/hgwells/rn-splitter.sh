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

# Roman numeral pattern for 1-20, can appear anywhere in the line
# I, II, III, IV, V, VI, VII, VIII, IX, X
# XI, XII, XIII, XIV, XV, XVI, XVII, XVIII, XIX, XX
roman_pattern="(I{1,3}|IV|V|VI{1,3}|IX|X|XI{1,3}|XIV|XV|XVI{1,3}|XIX|XX)\."

# Read the input file line by line
while IFS= read -r line || [[ -n "$line" ]]; do
    # Check if line contains a Roman numeral followed by a period
    if [[ "$line" =~ $roman_pattern ]]; then
        # If we were already in a scene, save the previous scene
        if $in_scene; then
            printf -v padded_count "%02d" $scene_count
            echo "$current_scene" > "${output_dir}/${padded_count}.txt"
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
    echo "$current_scene" > "${output_dir}/${padded_count}.txt"
fi

echo "Processing complete. Created $((scene_count)) scene files in the '$output_dir' directory."