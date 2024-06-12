locations = "mixed.txt"
output = "mixed2.txt"

with open(locations, 'r') as f_loc:
    with open(output, 'w') as f_out:
        timestamploceski = " "
        for line in f_loc:
            parts = line.split(' ')
            timestamploc = parts[0]

            if timestamploc == timestamploceski:
                time_parts = timestamploc.split(':')
                seconds, microseconds = time_parts[-1].split(',')
                microseconds = str(int(microseconds)+1)
                new = time_parts[0] + ":" +time_parts[1] + ":" + seconds + "," + microseconds + " " + parts[1] + " " + parts[2] + " " + parts[3]+ " " + parts[4]+ " " + parts[5] + "\n"
                f_out.write(new)
            else:
                f_out.write(line)
            
            timestamploceski = timestamploc
        