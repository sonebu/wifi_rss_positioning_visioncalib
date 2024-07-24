#!/bin/bash

INTERFACE="mon0"
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

file="$experiment_folder/tshark.json"

tshark -i mon0 -T json -e frame.time -e wlan_radio.signal_dbm -e wlan.sa -e wlan.ssid -e wlan.ra -e wlan.da -e wlan_radio.frequency -e wlan_radio.snr -e wlan_radio.noise_dbm -e wlan.antenna.id -l -j "frame" > "$file"

