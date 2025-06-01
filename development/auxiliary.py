# Description: This file contains the data dictionary for the evaluation data.
import math, torch, json, random
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
    
    #print(f"RSS normalization - Mean: {rss_mean}, Std: {rss_std}")
    #print(f"Location normalization - Mean: {loc_mean}, Std: {loc_std}")
    
    return inp_rss_vals_norm, gt_locations_norm, norm_params


def apply_normalization(inp_rss_vals, gt_locations, norm_params):
    # Extract normalization parameters
    rss_mean = norm_params['rss_mean']
    rss_std = norm_params['rss_std']
    loc_mean = norm_params['loc_mean']
    loc_std = norm_params['loc_std']
    
    # Apply normalization using the provided parameters
    inp_rss_vals_norm = (inp_rss_vals - rss_mean) / (rss_std + 1e-8)
    gt_locations_norm = (gt_locations - loc_mean) / (loc_std + 1e-8)
    
    #print(f"Applied normalization to new data:")
    #print(f"  RSS - Mean: {rss_mean}, Std: {rss_std}")
    #print(f"  Location - Mean: {loc_mean}, Std: {loc_std}")
    #print(f"  New RSS range: [{np.min(inp_rss_vals_norm):.3f}, {np.max(inp_rss_vals_norm):.3f}]")
    #print(f"  New Location range: [{np.min(gt_locations_norm):.3f}, {np.max(gt_locations_norm):.3f}]")
    
    # Check for potential issues
    #if np.abs(inp_rss_vals_norm).max() > 5:
    #    print(f"  ⚠️  WARNING: Normalized RSS values exceed ±5, may indicate domain shift!")
    #if np.abs(gt_locations_norm).max() > 5:
    #    print(f"  ⚠️  WARNING: Normalized location values exceed ±5, may indicate domain shift!")
    
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

def plot_rss_vs_position(inp_rss_vals, gt_locations, window_size=50):

    # Apply moving average filter to RSS values using 'valid' mode to avoid edge effects
    filtered_rss_0 = np.convolve(inp_rss_vals[:,0], np.ones(window_size)/window_size, mode='valid')
    filtered_rss_1 = np.convolve(inp_rss_vals[:,1], np.ones(window_size)/window_size, mode='valid')
    filtered_rss_2 = np.convolve(inp_rss_vals[:,2], np.ones(window_size)/window_size, mode='valid')
    
    filtered_rss = np.column_stack([filtered_rss_0, filtered_rss_1, filtered_rss_2])
    
    # Trim ground truth locations to match filtered RSS length
    # 'valid' mode reduces length by (window_size - 1)
    trim_start = window_size // 2
    trim_end = len(gt_locations) - (window_size - 1 - trim_start)
    gt_locations_trimmed = gt_locations[trim_start:trim_end]
    
    # Create time indices for the trimmed data
    time_indices = range(len(filtered_rss))
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'RSS and Position Over Time (Moving Average Window: {window_size})', fontsize=16)
    
    # Plot 1: RSS values and X position over time
    ax1 = axes[0]
    ax1.plot(time_indices, filtered_rss[:, 0], 'r-', alpha=0.7, label='RSS AP1')
    ax1.plot(time_indices, filtered_rss[:, 1], 'g-', alpha=0.7, label='RSS AP2')
    ax1.plot(time_indices, filtered_rss[:, 2], 'b-', alpha=0.7, label='RSS AP3')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('RSS (dBm)', color='black')
    ax1.set_title('RSS Values and X Position Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for X position
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_indices, gt_locations_trimmed[:, 0], 'k-', linewidth=1, label='X Position')
    ax1_twin.set_ylabel('X Position (m)', color='black')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: RSS values and Y position over time
    ax2 = axes[1]
    ax2.plot(time_indices, filtered_rss[:, 0], 'r-', alpha=0.7, label='RSS AP1')
    ax2.plot(time_indices, filtered_rss[:, 1], 'g-', alpha=0.7, label='RSS AP2')
    ax2.plot(time_indices, filtered_rss[:, 2], 'b-', alpha=0.7, label='RSS AP3')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('RSS (dBm)', color='black')
    ax2.set_title('RSS Values and Y Position Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Create second y-axis for Y position
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_indices, gt_locations_trimmed[:, 1], 'k-', linewidth=1, label='Y Position')
    ax2_twin.set_ylabel('Y Position (m)', color='black')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('rss_vs_position_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return filtered_rss, gt_locations_trimmed

def plot_sequential_results(results, num_clusters=13, window_size=50):
    # Handle CNN-only plotting when num_clusters is None
    plot_title = f'Sequential Inference Results (CNN Only)' if num_clusters is None else f'Sequential Inference Results (K-means clusters: {num_clusters})'
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    #fig.suptitle(plot_title, fontsize=16)
    
    # Extract final predictions for plotting (handle multi-timestep CNN outputs)
    cnn_preds = results['cnn_predictions']
    cnn_gts = results['cnn_ground_truths']

    cnn_preds_plot = cnn_preds
    cnn_gts_plot = cnn_gts
    
    # Get CNN data length
    cnn_len = len(results['cnn_errors'])
    
    # CNN predictions represent the MIDDLE of the window, not the end
    cnn_middle_offset = window_size // 2
    cnn_time = range(cnn_middle_offset, cnn_middle_offset + cnn_len)
    
    # Handle K-NN data only if available (num_clusters is not None)
    if num_clusters is not None and len(results['knn_interp_predictions']) > 0:
        knn_interp_preds = np.array(results['knn_interp_predictions'])
        knn_gts = np.array(results['knn_ground_truths'])
        knn_len = len(results['knn_interp_errors'])
        
        # Create properly aligned time indices
        knn_time = range(knn_len)
        
        # Extract K-NN data for the overlapping time period with CNN
        overlap_start = cnn_middle_offset
        overlap_end = min(cnn_middle_offset + cnn_len, knn_len)
        actual_overlap_len = overlap_end - overlap_start
        
        # Adjust CNN data to match available overlap
        if actual_overlap_len < cnn_len:
            cnn_preds_plot = cnn_preds_plot[:actual_overlap_len]
            cnn_gts_plot = cnn_gts_plot[:actual_overlap_len]
            cnn_time = range(cnn_middle_offset, overlap_end)
        
        # Extract overlapping K-NN data
        knn_interp_preds_overlap = knn_interp_preds[overlap_start:overlap_end]
        knn_gts_overlap = knn_gts[overlap_start:overlap_end]
        
        print(f"CNN time range: {min(cnn_time)} to {max(cnn_time)} ({len(cnn_time)} points)")
        print(f"K-NN time range: {min(knn_time)} to {max(knn_time)} ({len(knn_time)} points)")
        print(f"Overlap period: {overlap_start} to {overlap_end-1} ({actual_overlap_len} points)")
        print(f"CNN est. ofs.: {cnn_middle_offset} (window_size={window_size})")
        
        # Plot with K-NN data
        # X coordinates over time
        axes[0].plot(knn_time, knn_gts[:, 0], 'k-', label='Ground Truth', linewidth=1, alpha=0.8)
        axes[0].plot(cnn_time, cnn_preds_plot[:, 0, 0, 0], 'b-', label='CNN', alpha=0.9, linewidth=1)
        axes[0].plot(cnn_time, knn_interp_preds_overlap[:, 0], 'g-', label='K-NN+Interp', alpha=0.7, linewidth=1.5)
        axes[0].axvline(x=cnn_middle_offset, color='blue', linestyle=':', alpha=0.5, label='CNN Offset')
        
        # Y coordinates over time
        axes[1].plot(knn_time, knn_gts[:, 1], 'k-', label='Ground Truth', linewidth=1, alpha=0.8)
        axes[1].plot(cnn_time, cnn_preds_plot[:, 0, 1, 0], 'b-', label='CNN', alpha=0.9, linewidth=1)
        axes[1].plot(cnn_time, knn_interp_preds_overlap[:, 1], 'g-', label='K-NN+Interp', alpha=0.7, linewidth=1.5)
        axes[1].axvline(x=cnn_middle_offset, color='blue', linestyle=':', alpha=0.5, label='CNN Offset')
        
    else:
        # CNN-only plotting (no K-NN data available)
        print(f"CNN time range: {min(cnn_time)} to {max(cnn_time)} ({len(cnn_time)} points)")
        print(f"CNN est. ofs.: {cnn_middle_offset} (window_size={window_size})")
        print("K-NN plotting skipped (num_clusters is None or no K-NN data available)")
        
        # Plot CNN vs ground truth only
        # X coordinates over time
        axes[0].plot(cnn_time, cnn_gts_plot[:, 0, 0, 0], 'k-', label='GT', linewidth=1, alpha=0.8)
        axes[0].plot(cnn_time, cnn_preds_plot[:, 0, 0, 0], 'b-', label='Pred', alpha=0.9, linewidth=1)
        #axes[0].axvline(x=cnn_middle_offset, color='blue', linestyle=':', alpha=0.5, label='CNN Start')
        
        # Y coordinates over time
        axes[1].plot(cnn_time, cnn_gts_plot[:, 0, 1, 0], 'k-', label='GT', linewidth=1, alpha=0.8)
        axes[1].plot(cnn_time, cnn_preds_plot[:, 0, 1, 0], 'b-', label='Pred', alpha=0.9, linewidth=1)
        #axes[1].axvline(x=cnn_middle_offset, color='blue', linestyle=':', alpha=0.5, label='CNN Start')
    
    # Common plot settings
    axes[0].set_xlabel('time step')
    axes[0].set_ylabel('$x$ [m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('time step')
    axes[1].set_ylabel('$y$ [m]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sequential_inference_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_sequential_inference(sequential_loader, cnn_model, xts, yts, norm_params, 
                             RssPosAlgo_NearestNeighbour=None, RssPosAlgo_NearestNeighbour_Interpolation=None,
                             db_kmeans=None, db_is_normalized=True):
    cnn_predictions = []
    cnn_ground_truths = []
    knn_predictions = []
    knn_interp_predictions = []
    knn_ground_truths = []
    
    print(f"Running sequential inference on {len(sequential_loader)} windows...")
    
    # Process CNN data sequentially
    with torch.no_grad():
        for window_idx, (x_window, y_window) in enumerate(sequential_loader):
            # CNN prediction (CNN works on normalized data, outputs normalized predictions)
            cnn_pred_normalized = cnn_model(x_window.cuda())
            cnn_pred = denormalize_predictions(cnn_pred_normalized, norm_params)
            cnn_predictions.append(cnn_pred)
            
            # Handle ground truth - y_window should be normalized if coming from normalized dataset
            y_denorm = denormalize_predictions(y_window, norm_params)
            cnn_ground_truths.append(y_denorm)
    
    # Process K-NN data only if db_kmeans is provided
    if db_kmeans is not None:
        if RssPosAlgo_NearestNeighbour is None or RssPosAlgo_NearestNeighbour_Interpolation is None:
            raise ValueError("K-NN functions must be provided when db_kmeans is not None")
            
        print(f"Running K-NN inference on {len(xts)} samples...")
        for i, x_sample in enumerate(xts):
            # K-NN expects normalized RSS input if database was built with normalized RSS
            # But RSS normalization doesn't affect the coordinate outputs
            
            # K-NN predictions
            knn_pred = RssPosAlgo_NearestNeighbour(x_sample, db_kmeans)
            knn_interp_pred = RssPosAlgo_NearestNeighbour_Interpolation(x_sample, db_kmeans)
            
            # Handle K-NN prediction denormalization based on how database was built
            if db_is_normalized:
                # Database was built with normalized locations, so predictions are normalized
                knn_pred = denormalize_predictions(knn_pred, norm_params)
                knn_interp_pred = denormalize_predictions(knn_interp_pred, norm_params)
            # If db_is_normalized=False, predictions are already in original scale
            
            knn_predictions.append(knn_pred)
            knn_interp_predictions.append(knn_interp_pred)
            
            # Ground truth - yts should be normalized if from normalized dataset
            gt_denorm = denormalize_predictions(yts[i].numpy(), norm_params)
            knn_ground_truths.append(gt_denorm)
    else:
        print("Skipping K-NN inference (db_kmeans is None)")
    
    # Convert to numpy arrays
    results = {
        'cnn_predictions': np.array(cnn_predictions),
        'cnn_ground_truths': np.array(cnn_ground_truths),
        'knn_predictions': np.array(knn_predictions) if knn_predictions else np.array([]),
        'knn_interp_predictions': np.array(knn_interp_predictions) if knn_interp_predictions else np.array([]),
        'knn_ground_truths': np.array(knn_ground_truths) if knn_ground_truths else np.array([])
    }
    
    # Calculate errors
    results['cnn_errors'] = np.linalg.norm(results['cnn_ground_truths'] - results['cnn_predictions'], axis=1)
    
    if len(knn_predictions) > 0:
        results['knn_errors'] = np.linalg.norm(results['knn_ground_truths'] - results['knn_predictions'], axis=1)
        results['knn_interp_errors'] = np.linalg.norm(results['knn_ground_truths'] - results['knn_interp_predictions'], axis=1)
    else:
        results['knn_errors'] = np.array([])
        results['knn_interp_errors'] = np.array([])
    
    # Print performance summary
    print("\n" + "="*50)
    print("SEQUENTIAL INFERENCE PERFORMANCE SUMMARY")
    print("="*50)
    print(f"CNN Error: {np.mean(results['cnn_errors']):.3f} ± {np.std(results['cnn_errors']):.3f} m")
    
    if len(knn_predictions) > 0:
        print(f"K-NN Error: {np.mean(results['knn_errors']):.3f} ± {np.std(results['knn_errors']):.3f} m")
        print(f"K-NN+Interp Error: {np.mean(results['knn_interp_errors']):.3f} ± {np.std(results['knn_interp_errors']):.3f} m")
        print(f"K-NN samples: {len(results['knn_predictions'])}")
        print(f"Database built with {'normalized' if db_is_normalized else 'denormalized'} location data")
    else:
        print("K-NN evaluation skipped (db_kmeans was None)")
    
    print(f"CNN samples: {len(results['cnn_predictions'])}")
    
    return results