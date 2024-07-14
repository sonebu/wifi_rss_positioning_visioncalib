import binascii
import os
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
    parts = line.split(',')
    if len(parts) < 2:
        return line  # Skip lines that do not have all required parts

    timestamp = parts[0]
    time_parts = timestamp.split('.')
    
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
    parts[0] = new_timestamp
    return ','.join(parts)

def process_lines(input_file, output_intermediate_file):
    with open(input_file, 'r') as infile, open(output_intermediate_file, 'w') as outfile:
        for line in infile:
            splitted_line = line.split()
            if len(splitted_line) > 5:
                line = str(splitted_line[3]) + "," + ",".join(str(i) for i in splitted_line[5:])
                outfile.write(line + '\n')

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "../rss_value.txt")
    output_intermediate_file = os.path.join(base_dir, "processed_intermediate.txt")
    output_file = os.path.join(base_dir, "processed_rss.txt")
   
    process_lines(input_file, output_intermediate_file)
    
   
    with open(output_intermediate_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            new_line = process_line_timestamp(line.strip())
            outfile.write(new_line + '\n')
    