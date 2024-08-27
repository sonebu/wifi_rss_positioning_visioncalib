# Description: This file contains the data dictionary for the evaluation data.
import math
import torch
import json
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import random


def load_data(json_file):

    target_addresses=["d8:84:66:39:29:e8", "d8:84:66:03:61:00", "d8:84:66:4a:06:c9"]
    gt_locations = []
    inp_rss_vals = []
    with open(json_file) as f:
        data = json.load(f)

    second_hold = 5

    secondold = None
    rssi1 = None
    rssi2 = None
    rssi3 = None

    loc_x = None
    loc_y= None

    for item in data:
        
        timestampnew = item.get("_source", {}).get("layers", {}).get("frame.time", [])[0]
        secondnew = int(timestampnew.split(":")[-1].split(".")[0])
        #print(secondnew)
        loc_x_new = float(item.get("_source", {}).get("layers", {}).get("loc_x", [])[0])
        loc_y_new = float(item.get("_source", {}).get("layers", {}).get("loc_y", [])[0])


        # Get the source address and check if it's in the target addresses
        if item.get("_source", {}).get("layers", {}).get("wlan.sa", [])!=[]:
            source_address = item.get("_source", {}).get("layers", {}).get("wlan.sa", [])[0]
        else:
            continue
        
        if secondold is not None :
            if secondnew < secondold:
                secondnew += 60  # Adjust for rollover
        
            if (secondnew-secondold) > second_hold:
                secondold = None
                rssi1 = None
                rssi2 = None
                rssi3 = None

                continue

        # Extract the RSSI value
        if secondold is None or(secondnew - secondold) <= second_hold:
            #print("ts is ok")
            if source_address in target_addresses:
                rssi = float(item.get("_source", {}).get("layers", {}).get("wlan_radio.signal_dbm", [])[0])
                if source_address == target_addresses[0]:
                    rssi1 = rssi
                    #print("rssi1 is ok")
                elif source_address == target_addresses[2]:
                    rssi2 = rssi
                    #print("rssi2 is ok")
                elif source_address == target_addresses[1]:
                    rssi3 = rssi
                    #print("rssi3 is ok")

            if (rssi1 != None) and (rssi2 != None) and (rssi3 != None):
                inp_rss_vals.append([rssi1, rssi2, rssi3])  # We may have multiple RSSI values later, so keep it as a list
                gt_locations.append([loc_x, loc_y])
                loc_x = loc_x_new
                loc_y = loc_y_new
                secondold = secondnew % 60
                rssi1 = None
                rssi2 = None
                rssi3 = None
    
    # Zip the lists together
    combined = list(zip(inp_rss_vals[1:], gt_locations[1:]))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the combined list back into two separate lists
    inp_rss_vals_shuffled, gt_locations_shuffled = zip(*combined)

    # Convert back to lists (zip returns tuples)
    inp_rss_vals = list(inp_rss_vals_shuffled)
    gt_locations = list(gt_locations_shuffled)


    # Convert lists to numpy arrays
    gt_locations = np.asarray(gt_locations)
    inp_rss_vals = np.asarray(inp_rss_vals)
    return inp_rss_vals, gt_locations



def prepare_data_loaders(inp_rss_vals, gt_locations, batch_size=1, train_test_split=0.5):
    split_index = int(len(inp_rss_vals) * train_test_split)
    
    x_train = inp_rss_vals[:split_index, :]
    x_test = inp_rss_vals[split_index:, :]
    y_train = gt_locations[:split_index, :]
    y_test = gt_locations[split_index:, :]

    tensor_x_train = torch.tensor(x_train).float()
    tensor_y_train = torch.tensor(y_train).float()
    tensor_x_test = torch.tensor(x_test).float()
    tensor_y_test = torch.tensor(y_test).float()

    train_dataset = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=y_test.shape[0], shuffle=False)

    return train_loader, test_loader, tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test



data_dictionary = {
    '0_0': [-34, -63, -52],
    '0_7': [-50, -71, -70],
    '-6_7': [-45, -66, -66],
    '6_7': [-42, -65, -66],
    '-14_2': [-41, -71, -70],
    '14_2': [-48, -53, -53],
    '-28_-4': [-62, -71, -44],
    '-28_4': [-53, -73, -41],
    '28_-4': [-56, -44, -70],
    '28_4': [-53, -41, -72],
    '25_0': [-53, -36, -61],
    '-25_-2': [-54, -63, -37],
    '7_-2': [-42, -48, -56],
}




# placeholder