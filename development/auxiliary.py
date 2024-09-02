# Description: This file contains the data dictionary for the evaluation data.
import math, torch, json, random
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

### filters a data.json file for records from static targets (e.g., APs) which appear together (list of source addresses for static targets = target_addresses)
### there is a rolling-filter algorithm here with a "second_hold" time which, when it hits one of the target addresses, 
### it waits for the others to appear as well for a certain amount of time (hold).
### It drops that record if the others do not appear within the second_hold time
def loadData_staticTargetAddrMatch(json_file, second_hold = 5, shuffle=False, target_addresses=["d8:84:66:39:29:e8", "d8:84:66:4a:06:c9", "d8:84:66:03:61:00"]): 
    gt_locations = []
    inp_rss_vals = []
    with open(json_file) as f:
        data = json.load(f)

    # initialization
    secondold  = None
    rssi_list  = [None]*len(target_addresses)
    time_list  = [None]*len(target_addresses)
    loc_x_list = [None]*len(target_addresses)
    loc_y_list = [None]*len(target_addresses)

    # process each record in the dataset 
    for item in data:
        timestampnew = item.get("_source", {}).get("layers", {}).get("frame.time", [])[0] # note how only the seconds are extracted
        secondnew    = int(timestampnew.split(":")[-1].split(".")[0])
        loc_x_new    = float(item.get("_source", {}).get("layers", {}).get("loc_x", [])[0])
        loc_y_new    = float(item.get("_source", {}).get("layers", {}).get("loc_y", [])[0])

        # check if there is a source address, and extract it if there is
        if item.get("_source", {}).get("layers", {}).get("wlan.sa", [])!=[]:
            source_address = item.get("_source", {}).get("layers", {}).get("wlan.sa", [])[0]
        else:
            continue # skip this record if source address does not exist
        
        if secondold is not None :
            # this is to handle "minute-crossings", i.e., when the new record is in say 01:23:03.25 and old record is in 01:22:59.75
            # the +=60 is to make the new record later than the old one since the algorithm would fail otherwise. 
            if secondnew < secondold:
                secondnew += 60  # Adjust for rollover
        
            # there may be jumps in the recordings due to stop-restarts during experiments
            # this part filters those out (i.e., if some of the target APs do not appear within second_hold time, the record is dropped)
            if (secondnew-secondold) > second_hold:
                secondold  = None
                rssi_list  = [None]*len(target_addresses)
                time_list  = [None]*len(target_addresses)
                loc_x_list = [None]*len(target_addresses)
                loc_y_list = [None]*len(target_addresses)
                continue  # skip this record and reset the data lists

        # process the record if it's a valid one (i.e., if it has an src address, and if the time diff is not above the hold period.
        if secondold is None or(secondnew - secondold) <= second_hold:
            if source_address in target_addresses:
                rssi      = float(item.get("_source", {}).get("layers", {}).get("wlan_radio.signal_dbm", [])[0])

                # get index of the address in the target list so that we can populate that specific element in the data lists 
                addr_idx  = target_addresses.index(source_address)
                rssi_list[addr_idx]  = rssi
                time_list[addr_idx]  = secondnew
                loc_x_list[addr_idx] = loc_x_new
                loc_y_list[addr_idx] = loc_y_new
                
                ### Note: target address [1] and [2] were swapped earlier ( ["d8:84:66:39:29:e8", "d8:84:66:03:61:00", "d8:84:66:4a:06:c9"] ) 
                ###       and the if check was [2] = rssi2 and [1] = rssi3, I swapped the address list instead and switched [1] and [2] in the if check
                #if source_address == target_addresses[0]:
                #    rssi1 = rssi
                #elif source_address == target_addresses[1]:
                #    rssi2 = rssi
                #elif source_address == target_addresses[2]:
                #    rssi3 = rssi
            
            if(not(any(x is None for x in rssi_list))):
                inp_rss_vals.append(rssi_list)

                ### design decision: interpolate and get location at the average time point for the list
                ###                  e.g.,  >>> import numpy as np
                ###                         >>> time_list=np.asarray([10,12,30])
                ###                         >>> loc_x_list=np.asarray([1.0,1.5,3.0])
                ###                         >>> np.interp((time_list[0]+time_list[-1])/2,time_list,loc_x_list)
                ###                         2.1666666666666665
                ###                         >>> (time_list[0]+time_list[-1])/2
                ###                         20.0
                ###                         >>> 
                loc_x_interp = np.interp((time_list[0]+time_list[-1])/2, time_list, loc_x_list)
                loc_y_interp = np.interp((time_list[0]+time_list[-1])/2, time_list, loc_y_list)
                gt_locations.append([loc_x_interp, loc_y_interp])

                # reset
                secondold = secondnew % 60
                rssi_list  = [None]*len(target_addresses)
                time_list  = [None]*len(target_addresses)
                loc_x_list = [None]*len(target_addresses)
                loc_y_list = [None]*len(target_addresses)

            #if (rssi1 != None) and (rssi2 != None) and (rssi3 != None): # ToDo: change this to rssi_list
            #    inp_rss_vals.append([rssi1, rssi2, rssi3])  # We may have multiple RSSI values later, so keep it as a list
            #    gt_locations.append([loc_x, loc_y])
            #    secondold = secondnew % 60
            #    loc_x = loc_x_new 
            #    loc_y = loc_y_new
            #    rssi1 = None
            #    rssi2 = None
            #    rssi3 = None
    
    if(shuffle):
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

