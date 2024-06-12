import pandas as pd
import json

def mixer(file):
    # Read Excel file
    df = pd.read_excel(file)

    # Shuffle the rows
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Save the shuffled data back to Excel
    shuffled_df.to_excel('shuffled_excel_file.xlsx', index=False)


mixer("finaldata.xlsx")
# Read Excel file
df = pd.read_excel('shuffled_excel_file.xlsx')  
print(pd)
# Convert DataFrame to JSON
data = []
for index, row in df.iterrows():
    print(row)
    entry = {
        "timestamp": row["timestamp"],
        "location": {
            "loc_x": row["loc_x"],
            "loc_y": row["loc_y"]
        },
        "signal_strength": {
            "RSS_1": int(row["RSS_1"]),
            "RSS_2": int(row["RSS_2"]),
            "RSS_3": int(row["RSS_3"])
        }
    }
    data.append(entry)

json_data = {"data": data}

# Write JSON to file
with open("finaldata.json", "w") as f:
    json.dump(json_data, f, indent=4)