# Description: This file contains the data dictionary for the evaluation data.
import math, torch, json, random
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

### filters a data.json file for records from static targets (e.g., APs) which appear together (list of source addresses for static targets = target_addresses)
### there is a rolling-filter algorithm here with a "second_hold" time which, when it hits one of the target addresses, 
### it waits for the others to appear as well for a certain amount of time (hold).
### It drops that record if the others do not appear within the second_hold time
def loadData_staticTargetAddrMatch(json_file, second_hold = 5, shuffle=False, target_addresses=["d8:84:66:39:29:e8", "d8:84:66:4a:06:c9", "d8:84:66:03:61:00"], snap250ms=True): 
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
        loc_x_new    = item.get("_source", {}).get("layers", {}).get("loc_x", [])
        loc_y_new    = item.get("_source", {}).get("layers", {}).get("loc_y", [])
        if (snap250ms):
            loc_x_new = float(loc_x_new[0])
            loc_y_new = float(loc_y_new[0])
        else:
            loc_x_new = float(loc_x_new)
            loc_y_new = float(loc_y_new)

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

class SequentialDataset(Dataset):
    def __init__(self, inp_rss_vals, gt_locations, window_size, cnn_data=False, cnn_kernel_sizes=None):
        self.inp_rss_vals = inp_rss_vals
        self.gt_locations = gt_locations
        self.window_size = window_size
        self.cnn_data = cnn_data
        self.cnn_kernel_sizes = cnn_kernel_sizes
        
    def __len__(self):
        return len(self.inp_rss_vals) - self.window_size + 1

    def __getitem__(self, idx):
        if(self.cnn_data):
            x_seq = torch.swapaxes(torch.tensor(self.inp_rss_vals[idx:idx + self.window_size], dtype=torch.float32),0,1)
            y_seq = torch.swapaxes(torch.tensor(self.gt_locations[idx:idx + self.window_size], dtype=torch.float32),0,1)
            if(self.cnn_kernel_sizes is not None):
                total_reduction = sum(k - 1 for k in self.cnn_kernel_sizes)
                enlargement_kernel_size = total_reduction + 1
                enlargement = 1.0 / enlargement_kernel_size
                temp0 = torch.nn.functional.conv1d(y_seq[0,:].unsqueeze(0), torch.ones(1,1,enlargement_kernel_size)*enlargement, padding="valid")
                temp1 = torch.nn.functional.conv1d(y_seq[1,:].unsqueeze(0), torch.ones(1,1,enlargement_kernel_size)*enlargement, padding="valid")
                y_seq = torch.concat((temp0,temp1),dim=0) 
        else:
            x_seq = torch.tensor(self.inp_rss_vals[idx:idx + self.window_size], dtype=torch.float32)
            y_seq = torch.tensor(self.gt_locations[idx:idx + self.window_size], dtype=torch.float32)
        return x_seq, y_seq

class RandomizedSequentialDataset(Dataset):
    def __init__(self, inp_rss_vals, gt_locations, window_size, window_indices, cnn_data=False, cnn_kernel_sizes=None):
        self.inp_rss_vals = inp_rss_vals
        self.gt_locations = gt_locations
        self.window_size = window_size
        self.window_indices = window_indices  # Pre-computed randomized window start indices
        self.cnn_data = cnn_data
        self.cnn_kernel_sizes = cnn_kernel_sizes
        
    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        # Use the pre-computed randomized window start index
        window_start_idx = self.window_indices[idx]
        
        if(self.cnn_data):
            x_seq = torch.swapaxes(torch.tensor(self.inp_rss_vals[window_start_idx:window_start_idx + self.window_size], dtype=torch.float32),0,1)
            y_seq = torch.swapaxes(torch.tensor(self.gt_locations[window_start_idx:window_start_idx + self.window_size], dtype=torch.float32),0,1)
            if(self.cnn_kernel_sizes is not None):
                total_reduction = sum(k - 1 for k in self.cnn_kernel_sizes)
                enlargement_kernel_size = total_reduction + 1
                enlargement = 1.0 / enlargement_kernel_size
                temp0 = torch.nn.functional.conv1d(y_seq[0,:].unsqueeze(0), torch.ones(1,1,enlargement_kernel_size)*enlargement, padding="valid")
                temp1 = torch.nn.functional.conv1d(y_seq[1,:].unsqueeze(0), torch.ones(1,1,enlargement_kernel_size)*enlargement, padding="valid")
                y_seq = torch.concat((temp0,temp1),dim=0) 
        else:
            x_seq = torch.tensor(self.inp_rss_vals[window_start_idx:window_start_idx + self.window_size], dtype=torch.float32)
            y_seq = torch.tensor(self.gt_locations[window_start_idx:window_start_idx + self.window_size], dtype=torch.float32)
        return x_seq, y_seq

def prepare_data_loaders(inp_rss_vals, gt_locations, batch_size=1, window_size=20, train_test_split=0.5, cnn_data=False, cnn_kernel_sizes=None, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate total number of possible windows
    total_windows = len(inp_rss_vals) - window_size + 1
    
    # Create all possible window start indices
    all_window_indices = np.arange(total_windows)
    
    # Randomly shuffle the window indices
    np.random.shuffle(all_window_indices)
    
    # Split the window indices into train and test
    split_index = int(total_windows * train_test_split)
    train_window_indices = all_window_indices[:split_index]
    test_window_indices = all_window_indices[split_index:]
    
    # Create tensor versions for K-NN methods (using original sequential split for compatibility)
    sample_split_index = int(len(inp_rss_vals) * train_test_split)
    tensor_x_train = torch.tensor(inp_rss_vals[:sample_split_index, :], dtype=torch.float32)
    tensor_y_train = torch.tensor(gt_locations[:sample_split_index, :], dtype=torch.float32)
    tensor_x_test = torch.tensor(inp_rss_vals[sample_split_index:, :], dtype=torch.float32)
    tensor_y_test = torch.tensor(gt_locations[sample_split_index:, :], dtype=torch.float32)

    # Create datasets with randomized windows
    train_dataset = RandomizedSequentialDataset(inp_rss_vals, gt_locations, window_size, train_window_indices, cnn_data, cnn_kernel_sizes)
    test_dataset = RandomizedSequentialDataset(inp_rss_vals, gt_locations, window_size, test_window_indices, cnn_data, cnn_kernel_sizes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test


def create_sequential_inference_loader(inp_rss_vals, gt_locations, window_size, cnn_data=True, cnn_kernel_sizes=None):
    """
    Create a DataLoader that simulates real-world sequential data flow
    Returns one window at a time in chronological order
    Also returns tensor versions of the input data for K-NN methods
    """
    # Create the sequential dataset
    dataset = SequentialDataset(inp_rss_vals, gt_locations, window_size, cnn_data, cnn_kernel_sizes)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create tensor versions for K-NN methods (same as prepare_data_loaders)
    tensor_x = torch.tensor(inp_rss_vals, dtype=torch.float32)
    tensor_y = torch.tensor(gt_locations, dtype=torch.float32)
    
    return loader, tensor_x, tensor_y


def normalize_rss_data(inp_rss_vals, gt_locations, fit_on_train=True, train_split=0.8):
    if fit_on_train:
        # Calculate normalization parameters only on training data
        split_idx = int(len(inp_rss_vals) * train_split)
        rss_train = inp_rss_vals[:split_idx]
        loc_train = gt_locations[:split_idx]
        
        # RSS normalization parameters
        rss_mean = np.mean(rss_train, axis=0)
        rss_std = np.std(rss_train, axis=0)
        
        # Location normalization parameters  
        loc_mean = np.mean(loc_train, axis=0)
        loc_std = np.std(loc_train, axis=0)
    else:
        # Use all data for normalization (less ideal)
        rss_mean = np.mean(inp_rss_vals, axis=0)
        rss_std = np.std(inp_rss_vals, axis=0)
        loc_mean = np.mean(gt_locations, axis=0)
        loc_std = np.std(gt_locations, axis=0)
    
    # Normalize the data
    inp_rss_vals_norm = (inp_rss_vals - rss_mean) / (rss_std + 1e-8)  # Add small epsilon to avoid division by zero
    gt_locations_norm = (gt_locations - loc_mean) / (loc_std + 1e-8)
    
    # Return normalized data and parameters for denormalization
    norm_params = {
        'rss_mean': rss_mean, 'rss_std': rss_std,
        'loc_mean': loc_mean, 'loc_std': loc_std
    }
    
    print(f"RSS normalization - Mean: {rss_mean}, Std: {rss_std}")
    print(f"Location normalization - Mean: {loc_mean}, Std: {loc_std}")
    
    return inp_rss_vals_norm, gt_locations_norm, norm_params


def denormalize_predictions(predictions, norm_params):
    # Convert different input types to numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    elif isinstance(predictions, (tuple, list)):
        predictions = np.array(predictions)
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Handle different prediction shapes
    if len(predictions.shape) == 3:  # Shape: (batch, coordinates, time_steps)
        # Reshape normalization parameters to match prediction dimensions
        loc_mean = norm_params['loc_mean'].reshape(1, -1, 1)  # (1, 2, 1)
        loc_std = norm_params['loc_std'].reshape(1, -1, 1)    # (1, 2, 1)
    elif len(predictions.shape) == 2:  # Shape: (batch, coordinates) or (coordinates, time_steps)
        if predictions.shape[0] == 2 and predictions.shape[1] > 2:  # (coordinates, time_steps)
            loc_mean = norm_params['loc_mean'].reshape(-1, 1)  # (2, 1)
            loc_std = norm_params['loc_std'].reshape(-1, 1)    # (2, 1)
        else:  # (batch, coordinates)
            loc_mean = norm_params['loc_mean']  # (2,)
            loc_std = norm_params['loc_std']    # (2,)
    else:  # Shape: (coordinates,)
        loc_mean = norm_params['loc_mean']
        loc_std = norm_params['loc_std']
    
    return predictions * loc_std + loc_mean


def denormalize_rss(rss_normalized, norm_params):
    if isinstance(rss_normalized, torch.Tensor):
        rss_normalized = rss_normalized.detach().cpu().numpy()
    
    return rss_normalized * norm_params['rss_std'] + norm_params['rss_mean']
