import json
import os
import argparse
from collections import defaultdict
def round_milliseconds(msec):
    if msec < 13:
        return 0
    elif msec < 38:
        return 25
    elif msec < 63:
        return 50
    elif msec < 88:
        return 75
    else:
        return 0

def process_line_timestamp(line):


    
    time_parts = line.split('.')
    
    if len(time_parts) != 2:
        return line  # Skip lines that do not have the expected timestamp format

    hms, usec = time_parts
    msec = int(usec[:2])  # Keep only the first 3 digits
    
    rounded_msec = round_milliseconds(msec)
    
    # Handle overflow in seconds
    if msec >= 88:
        h, m, s = map(int, hms.split(':'))
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
        hms = f"{h:02}:{m:02}:{s:02}"
        rounded_msec = 0

    new_timestamp = f"{hms}.{rounded_msec:02d}"
    
    return new_timestamp


def hex_to_ascii(hex_str):
    # Remove any non-hexadecimal characters and spaces
    cleaned_hex_str = ''.join(c for c in hex_str if c in '0123456789abcdefABCDEF')
    
    # Check if the cleaned string is of even length
    if len(cleaned_hex_str) % 2 != 0:
        raise ValueError("Hexadecimal string length must be even.")
    
    try:
        # Convert hex to bytes
        bytes_object = bytes.fromhex(cleaned_hex_str)
        # Convert bytes to ASCII string
        ascii_str = bytes_object.decode('ascii', errors='ignore')
        return ascii_str
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_line_loc_xy(line):
    parts = line.split(',')
    if len(parts) <= 2:
        return line  # Skip lines that do not have all required parts

    timestamp = parts[0]
    time_parts = timestamp.split('.')
    if len(time_parts) != 2:
        return line  # Skip lines that do not have the expected timestamp format

    hms, usec = time_parts
    msec = int(usec[:2])  # Keep only the first 3 digits
    

    rounded_msec = round_milliseconds(msec)
    if msec >= 88:
        # Add 1 to the second part and reset milliseconds to 00
        hms_parts = hms.split(':')
        h, m, s = map(int, hms_parts)
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
        hms = f"{h:02}:{m:02}:{s:02}"
        rounded_msec = 0

    new_timestamp = f"{hms}.{rounded_msec:02d}"

    parts[0] = new_timestamp
    return ','.join(parts)

def main(experiment_folder):
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path=str(experiment_folder)+"/loc_xy.txt"
    input_file = os.path.join(base_dir, path)
    output_file = os.path.join(base_dir, "processed_loc_xy.txt")  # Example output file

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            new_line = process_line_loc_xy(line.strip())
            outfile.write(new_line + '\n')

def transform_timestamp_json(input_file):
    # Open and read the input JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    # Prepare the transformed data
    transformed_data = []
    for item in data:
        frame_time = item.get("_source", {}).get("layers", {}).get("frame.time", [""])[0]
        time_only = frame_time.split(" ")[4] if " " in frame_time else frame_time
        time=process_line_timestamp(time_only)
        wlan_ssid = item.get("_source", {}).get("layers", {}).get("wlan.ssid", [""])[0]
        wlan_ssid=hex_to_ascii(wlan_ssid)
        transformed_item = {
            "_index": item.get("_index", "unknown"),
            "_type": item.get("_type", "doc"),
            "_score": item.get("_score", None),
            "_source": {
                "layers": {
                    "frame.time": [time],
                    "wlan_radio.signal_dbm": item.get("_source", {}).get("layers", {}).get("wlan_radio.signal_dbm", []),
                    "wlan.sa": item.get("_source", {}).get("layers", {}).get("wlan.sa", []),
                    "wlan.ra": item.get("_source", {}).get("layers", {}).get("wlan.ra", []),
                    "wlan.da": item.get("_source", {}).get("layers", {}).get("wlan.da", []),
                    "wlan_radio.frequency": item.get("_source", {}).get("layers", {}).get("wlan_radio.frequency", []),
                    "wlan.ssid": [wlan_ssid]
                }
            }
        }
        transformed_data.append(transformed_item)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "processed_rss.json")  # Example output file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=2)

def read_location_data(file_path):
    location_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                timestamp, loc_x, loc_y = parts
                location_data.append((timestamp, loc_x, loc_y))
    return location_data

def update_json_with_location(experiment_folder):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, "processed_rss.json")
    location_file = os.path.join(base_dir, "processed_loc_xy.txt")
    path=str(experiment_folder)+"/data.json"
    output_file = os.path.join(base_dir, path)  # Example output file
    location_data = read_location_data(location_file)
    

    # Create a dictionary for fast lookup
    location_dict = {}
    for timestamp, loc_x, loc_y in location_data:
        if timestamp not in location_dict:
            location_dict[timestamp] = []
        location_dict[timestamp].append((loc_x, loc_y))
    
    # Read and update the JSON file
    with open(json_file, 'r') as infile:
        data = json.load(infile)
    
    for item in data:
        frame_time = item.get("_source", {}).get("layers", {}).get("frame.time", [""])[0]
        # Find matching location data
       
        if frame_time in location_dict and loc_x not in item["_source"].get("layers", {}):
            
            # Assuming each timestamp could have multiple loc_x and loc_y pairs
            locs = location_dict[frame_time]
            print(locs)
            item["_source"]["layers"]["loc_x"] = [locs[0][0]]
            item["_source"]["layers"]["loc_y"] = [locs[0][1]]
    
    # Write the updated JSON to a new file
    filtered_data = []
    for item in data:
        layers = item["_source"].get("layers", {})
        signal_dbm = layers.get("wlan_radio.signal_dbm", [])
        if "loc_x" in layers and "loc_y" in layers and signal_dbm: 
            filtered_data.append(item)
    
    with open(output_file, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=2)
def remove_files():
    files_to_remove = ['processed_loc_xy.txt', 'processed_rss.json']
    
    for file_name in files_to_remove:
        try:
            os.remove(file_name)
           
        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except PermissionError:
            print(f"Permission denied: {file_name}")
        except Exception as e:
            print(f"Error removing {file_name}: {e}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment folder.')
    parser.add_argument('-e', '--experiment', required=True, help='Path to the experiment folder')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, args.experiment+"/tshark.json" )
    
    
    main(args.experiment)
    transform_timestamp_json(input_file)
    update_json_with_location(args.experiment)
    remove_files()
