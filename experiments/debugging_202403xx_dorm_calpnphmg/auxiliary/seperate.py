def round_timestamp(timestamp):
    # Split timestamp into hours, minutes, seconds, and microseconds
    time_parts = timestamp.split(':')
    seconds, microseconds = time_parts[-1].split(',')

    # Round seconds to the nearest 25
    rounded_microseconds = str(int(round(int(microseconds) / 250000000) * 25))

    # If rounded seconds is 100, increment minutes and set seconds to 00
    if rounded_microseconds == '100':
        rounded_microseconds = '00'
        seconds = str(int(seconds) + 1).zfill(2)
    elif rounded_microseconds == '0':
        rounded_microseconds = "00"


    # Format the rounded timestamp
    rounded_timestamp = time_parts[0] + ":" +time_parts[1] + ":" + seconds + "," + rounded_microseconds
    return rounded_timestamp

# Read input file5  5   5
input_file = "kayit4.txt"
output_file = "kayit4ro.txt"

with open(input_file, 'r') as f_in:
    with open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.split('\t')
            timestamp = parts[0]
            rounded_timestamp = round_timestamp(timestamp)
            modified_line = rounded_timestamp + "  " + parts[1] + "  " + parts[2] + "  " + parts[3]
            modified_line = modified_line 
            f_out.write(modified_line)

print("Timestamps rounded and written to output file.")