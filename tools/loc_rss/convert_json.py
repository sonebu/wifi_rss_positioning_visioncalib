import json, os

def convert_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as out_f:
            for line in f:
                parts = line.strip().split(';')
                timestamp = parts[0]
                source_id = parts[1]
                locx = float(parts[2])
                locy = float(parts[3])
                rss = parts[4].strip("[]").split(",")  # Keep as list

                # Clean up any extra whitespace around RSS values
                rss = [value.strip() for value in rss]
                
                json_data = {
                    "timestamp": timestamp,
                    "source_id": source_id,
                    "locx": locx,
                    "locy": locy,
                    "rss": rss
                }
                out_f.write(json.dumps(json_data) + '\n')



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "merged_output.txt" )
    output_file = os.path.join(base_dir,"../output.json")

    convert_to_json(input_file, output_file)
