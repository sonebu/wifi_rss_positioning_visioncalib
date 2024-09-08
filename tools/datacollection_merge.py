import json, os, argparse
import numpy as np
from collections import defaultdict

def process_line_timestamp(line):
    time_parts = line.split('.')
    if len(time_parts) != 2:
        return line  # Skip lines that do not have the expected timestamp format

    hms, usec = time_parts
    msec = int(usec[:3])  # Keep only the first 3 digits
    
    rounded_msec = msec # skipping a rounding function here
    if msec >= 999:
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

    new_timestamp = f"{hms}.{rounded_msec:03d}"
    
    return new_timestamp

def process_line_locxy(line):
    parts = line.split(',')
    if len(parts) <= 2:
        return line  # Skip lines that do not have all required parts
    timestamp = parts[0]
    new_timestamp = process_line_timestamp(timestamp)
    parts[0] = new_timestamp
    return ','.join(parts)

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

def process_locxy(experiment_folder):
    input_file  = experiment_folder+"loc_xy.txt"
    output_file = experiment_folder+"processed_loc_xy.txt"  # Example output file

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            new_line = process_line_locxy(line.strip())
            outfile.write(new_line + '\n')

def transform_timestamp_json(experiment_folder):
    input_file = experiment_folder+"tshark.json"
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

    output_file = experiment_folder+"processed_rss.json"
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

def remove_files(experiment_folder):
    files_to_remove = [experiment_folder+'processed_loc_xy.txt', 
                       experiment_folder+'processed_rss.json']
    
    for file_name in files_to_remove:
        try:
            os.remove(file_name)
           
        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except PermissionError:
            print(f"Permission denied: {file_name}")
        except Exception as e:
            print(f"Error removing {file_name}: {e}")

def frametime2secs(frametime):
    hrs   = int(frametime.split(":")[0])
    mins  = int(frametime.split(":")[1])
    secs  = int((frametime.split(":")[2]).split(".")[0])
    msecs = int((frametime.split(":")[2]).split(".")[1])
    total_seconds = (hrs*60*60*1000 + mins*60*1000 + secs*1000 + msecs) / 1000.0 
    return total_seconds

def get_timeinsecs_rss(sampledict, offset=0):
    return frametime2secs(sampledict["_source"]["layers"]["frame.time"][0]) - offset;

def update_json_with_location(experiment_folder, verbose=False):
    rss_file    = experiment_folder+"processed_rss.json"
    loc_file    = experiment_folder+"processed_loc_xy.txt"
    output_file = experiment_folder+"data.json"
    
    location_data = read_location_data(loc_file)
    location_time = np.zeros(len(location_data))
    for i, loc_item in enumerate(location_data):
        location_time[i] = frametime2secs(location_data[i][0]) # [0] --> first item of tuple is timestamp
    loc_starttime = location_time[0]
    
    # Read and update the JSON file
    with open(rss_file, 'r') as infile:
        data = json.load(infile)
    
    rss_starttime = get_timeinsecs_rss(data[0])
    firststartsec = min(rss_starttime, loc_starttime)
    location_time_seconds_suboffset = location_time - firststartsec
    
    for item in data:
        rssitem_time_seconds_suboffset = get_timeinsecs_rss(item,offset=firststartsec)
        time_distance_between_rss_and_closest_loc = np.abs(location_time_seconds_suboffset - rssitem_time_seconds_suboffset)
        closest_location_idx = np.argmin(time_distance_between_rss_and_closest_loc)
        closest_location_time = location_time_seconds_suboffset[closest_location_idx]
        closest_location = (location_data[closest_location_idx][1], location_data[closest_location_idx][2])
        if(verbose):
            print(rssitem_time_seconds_suboffset, closest_location_time, closest_location)            
        
        ### NOTE: this is a 2.0 second waiting time, a "hold" mechanism 
        ###       when we are hopping between 2 channels with 0.5s dwell this is fine,
        ###       but when we hop between many channels with longer dwell times, 
        ###       this will probably fail data collection altogeher
        if(np.min(time_distance_between_rss_and_closest_loc) < 2.0):
            item["_source"]["layers"]["loc_x"] = closest_location[0]
            item["_source"]["layers"]["loc_y"] = closest_location[1]
        else:
            continue # skip this sample if it hits the hold mechanism

    
    # Write the updated JSON to a new file
    filtered_data = []
    for item in data:
        layers = item["_source"].get("layers", {})
        signal_dbm = layers.get("wlan_radio.signal_dbm", [])
        if "loc_x" in layers and "loc_y" in layers and signal_dbm: 
            filtered_data.append(item)
    
    with open(output_file, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment folder.')
    parser.add_argument('-e', '--experiment', required=True, help='Path to the experiment folder')
    args = parser.parse_args()

    expfolder = args.experiment

    process_locxy(expfolder)
    transform_timestamp_json(expfolder)
    update_json_with_location(expfolder)
    remove_files(expfolder)
