import os
import argparse
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


def process_line(line):
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
            new_line = process_line(line.strip())
            outfile.write(new_line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment folder.')
    parser.add_argument('-e', '--experiment', required=True, help='Path to the experiment folder')

    args = parser.parse_args()
    
    
    main(args.experiment)