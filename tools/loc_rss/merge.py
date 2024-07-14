from collections import defaultdict
import os
def merge_files(file1, file2, output_file):
    # Read the first file and store source ID and RSS by timestamp
    data1 = defaultdict(lambda: defaultdict(list))
    with open(file1, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
            if len(parts) > 2:
                
                timestamp = parts[0]
                source_id = parts[2]
                
                data1[timestamp][source_id].append(parts[1])
            else:
                print(parts)
    
    # Read the second file and store locx and locy by timestamp
    data2 = defaultdict(dict)
    with open(file2, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            timestamp = parts[0]
            locx, locy = parts[1:3]
            
            data2[timestamp] = (locx, locy)

    # Combine the results into the final output
    merged_data = []
   
    for timestamp in sorted(data1.keys()):
        if timestamp in data2:
            locx, locy = data2[timestamp]
            
            for source_id, rss_list in data1[timestamp].items():
                
                merged_data.append(f"{timestamp};{source_id};{locx};{locy};{rss_list}")
                
    # Write the merged results to the output file
    with open(output_file, 'w') as f:
        for line in merged_data:
            f.write(f"{line}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file1 =  os.path.join(base_dir,"processed_rss.txt")  # Replace with your first input file
    file2 =  os.path.join(base_dir,"processed_loc_xy.txt")  # Replace with your second input file
    output_file = os.path.join(base_dir,"merged_output.txt")
    merge_files(file1, file2, output_file)