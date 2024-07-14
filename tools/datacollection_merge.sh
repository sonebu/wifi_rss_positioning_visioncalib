#!/bin/bash

#
experiment_folder=""

# Function to display usage
usage() {
    echo "Usage: $0 -e <experiment_folder>"
    exit 1
}

# Parse command-line options
while getopts "e:" opt; do
    case $opt in
        e)
            experiment_folder="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done

# Check if the experiment folder is provided
if [ -z "$experiment_folder" ]; then
    usage
fi

# Output the stored experiment folder


python loc_rss/loc_merge.py -e $experiment_folder
python loc_rss/rss_merge.py
python loc_rss/merge.py
python loc_rss/convert_json.py


rm loc_rss/merged_output.txt loc_rss/processed_intermediate.txt loc_rss/processed_loc_xy.txt loc_rss/processed_rss.txt