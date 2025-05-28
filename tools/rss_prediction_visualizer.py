import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Add development directory to path for importing algorithms and auxiliary
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'development'))
from auxiliary import loadData_staticTargetAddrMatch, prepare_data_loaders
from algorithms import (
    RssPosAlgo_NearestNeighbour,
    RssPosAlgo_NearestNeighbour_Interpolation,
    RssPosAlgo_NearestNeighbour_GetKmeansDb,
    RssPosAlgo_NeuralNet_CNNv1
)

# Hardcoded target addresses
TARGET_ADDRESSES = ["d8:47:32:eb:6c:38", "50:c7:bf:19:e6:4d", "18:28:61:3d:94:7a"]


def load_cnn_model(model_path):
    """Load a trained CNN model."""
    # Use default architecture from the notebook
    kernel_sizes = [13, 9, 5, 3]
    model = RssPosAlgo_NeuralNet_CNNv1([8, 16, 8, 2], kernel_sizes)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def generate_predictions(inp_rss_vals, gt_locations, cnn_model, num_clusters):
    """Generate predictions using both CNN and nearest neighbor methods."""
    
    # For CNN, we need to prepare data with sequences
    kernel_sizes = [13, 9, 5, 3]
    window_size = 50
    batch_size = 1
    train_test_split = 0.8  # Use 80% for training (for K-means), 20% for testing
    
    print("Preparing CNN data...")
    # Prepare data loaders for CNN
    _, test_loader, xtr, ytr, xts, yts = prepare_data_loaders(
        inp_rss_vals, gt_locations,
        batch_size=batch_size, window_size=window_size,
        train_test_split=train_test_split,
        cnn_data=True, cnn_kernel_sizes=kernel_sizes
    )
    
    print(f"Building K-means database with {num_clusters} clusters...")
    # Use torch tensors for K-means (as in the notebook)
    db_kmeans = RssPosAlgo_NearestNeighbour_GetKmeansDb(xtr, ytr, num_clusters=num_clusters)
    
    print("Generating CNN predictions...")
    cnn_predictions = []
    # Generate CNN predictions
    for test_inputs, test_labels in test_loader:
        with torch.no_grad():
            loc_pred_cnn = cnn_model(test_inputs)
            loc_pred_cnn_np = loc_pred_cnn.detach().numpy()
            # CNN outputs sequences, take the last prediction
            cnn_predictions.append(loc_pred_cnn_np[0, :, -1])
    
    print("Generating nearest neighbor predictions...")
    nearest_neighbor_predictions = []
    nearest_neighbor_interp_predictions = []
    
    # For nearest neighbor, use the test data (torch tensors as in notebook)
    for i, x_test_sample in enumerate(xts):
        if i % 100 == 0:
            print(f"Processing NN sample {i}/{len(xts)}")
        
        knn_pred = RssPosAlgo_NearestNeighbour(x_test_sample, db_kmeans)
        knn_interp_pred = RssPosAlgo_NearestNeighbour_Interpolation(x_test_sample, db_kmeans)
        
        nearest_neighbor_predictions.append(knn_pred)
        nearest_neighbor_interp_predictions.append(knn_interp_pred)
    
    # Adjust ground truth to match CNN output length
    if isinstance(yts, torch.Tensor):
        gt_locations_final = yts.numpy()
    else:
        gt_locations_final = yts
    
    return (np.array(cnn_predictions), 
            np.array(nearest_neighbor_predictions), 
            np.array(nearest_neighbor_interp_predictions),
            gt_locations_final)


def create_plots(gt_locations, cnn_predictions, nearest_neighbor_predictions, 
                nearest_neighbor_interp_predictions, output_folder):
    """Create comparison plots between ground truth and predictions."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure all arrays have the same length (CNN predictions determine the length due to windowing)
    min_length = min(len(gt_locations), len(cnn_predictions), 
                     len(nearest_neighbor_predictions), len(nearest_neighbor_interp_predictions))
    
    gt_locations = gt_locations[:min_length]
    cnn_predictions = cnn_predictions[:min_length]
    nearest_neighbor_predictions = nearest_neighbor_predictions[:min_length]
    nearest_neighbor_interp_predictions = nearest_neighbor_interp_predictions[:min_length]
    
    print(f"Using {min_length} samples for plotting (aligned to CNN output length)")
    
    # Calculate errors
    cnn_errors = np.array([np.linalg.norm(gt_locations[i] - cnn_predictions[i]) 
                          for i in range(len(gt_locations))])
    knn_errors = np.array([np.linalg.norm(gt_locations[i] - nearest_neighbor_predictions[i]) 
                          for i in range(len(gt_locations))])
    knn_interp_errors = np.array([np.linalg.norm(gt_locations[i] - nearest_neighbor_interp_predictions[i]) 
                                 for i in range(len(gt_locations))])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RSS-based Location Prediction Analysis', fontsize=16)
    
    # Plot 1: X coordinates over time
    axes[0, 0].plot(range(len(gt_locations)), gt_locations[:, 0], 'g-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot(range(len(cnn_predictions)), cnn_predictions[:, 0], 'b--', label='CNN', alpha=0.7)
    axes[0, 0].plot(range(len(nearest_neighbor_predictions)), nearest_neighbor_predictions[:, 0], 'r:', label='K-NN', alpha=0.7)
    axes[0, 0].plot(range(len(nearest_neighbor_interp_predictions)), nearest_neighbor_interp_predictions[:, 0], 'm-.', label='K-NN + Interpolation', alpha=0.7)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('X Coordinate (m)')
    axes[0, 0].set_title('X Coordinate Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y coordinates over time
    axes[0, 1].plot(range(len(gt_locations)), gt_locations[:, 1], 'g-', label='Ground Truth', linewidth=2)
    axes[0, 1].plot(range(len(cnn_predictions)), cnn_predictions[:, 1], 'b--', label='CNN', alpha=0.7)
    axes[0, 1].plot(range(len(nearest_neighbor_predictions)), nearest_neighbor_predictions[:, 1], 'r:', label='K-NN', alpha=0.7)
    axes[0, 1].plot(range(len(nearest_neighbor_interp_predictions)), nearest_neighbor_interp_predictions[:, 1], 'm-.', label='K-NN + Interpolation', alpha=0.7)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Y Coordinate (m)')
    axes[0, 1].set_title('Y Coordinate Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 2D trajectory
    axes[1, 0].plot(gt_locations[:, 0], gt_locations[:, 1], 'g-', label='Ground Truth', linewidth=2)
    axes[1, 0].scatter(cnn_predictions[:, 0], cnn_predictions[:, 1], c='blue', alpha=0.6, s=20, label='CNN')
    axes[1, 0].scatter(nearest_neighbor_predictions[:, 0], nearest_neighbor_predictions[:, 1], c='red', alpha=0.6, s=20, label='K-NN')
    axes[1, 0].scatter(nearest_neighbor_interp_predictions[:, 0], nearest_neighbor_interp_predictions[:, 1], c='magenta', alpha=0.6, s=20, label='K-NN + Interpolation')
    axes[1, 0].set_xlabel('X Coordinate (m)')
    axes[1, 0].set_ylabel('Y Coordinate (m)')
    axes[1, 0].set_title('2D Trajectory Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Plot 4: Error comparison
    axes[1, 1].plot(range(len(cnn_errors)), cnn_errors, 'b-', 
                   label=f'CNN (μ={np.mean(cnn_errors):.3f}±{np.std(cnn_errors):.3f})', alpha=0.7)
    axes[1, 1].plot(range(len(knn_errors)), knn_errors, 'r-', 
                   label=f'K-NN (μ={np.mean(knn_errors):.3f}±{np.std(knn_errors):.3f})', alpha=0.7)
    axes[1, 1].plot(range(len(knn_interp_errors)), knn_interp_errors, 'm-', 
                   label=f'K-NN + Interp (μ={np.mean(knn_interp_errors):.3f}±{np.std(knn_interp_errors):.3f})', alpha=0.7)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Prediction Error (m)')
    axes[1, 1].set_title('Prediction Error Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_folder, 'rss_prediction_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()


def save_results_to_csv(gt_locations, cnn_predictions, nearest_neighbor_predictions, 
                       nearest_neighbor_interp_predictions, output_folder):
    """Save prediction results to CSV file."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure all arrays have the same length (CNN predictions determine the length due to windowing)
    min_length = min(len(gt_locations), len(cnn_predictions), 
                     len(nearest_neighbor_predictions), len(nearest_neighbor_interp_predictions))
    
    gt_locations = gt_locations[:min_length]
    cnn_predictions = cnn_predictions[:min_length]
    nearest_neighbor_predictions = nearest_neighbor_predictions[:min_length]
    nearest_neighbor_interp_predictions = nearest_neighbor_interp_predictions[:min_length]
    
    # Calculate errors
    cnn_errors = np.array([np.linalg.norm(gt_locations[i] - cnn_predictions[i]) 
                          for i in range(len(gt_locations))])
    knn_errors = np.array([np.linalg.norm(gt_locations[i] - nearest_neighbor_predictions[i]) 
                          for i in range(len(gt_locations))])
    knn_interp_errors = np.array([np.linalg.norm(gt_locations[i] - nearest_neighbor_interp_predictions[i]) 
                                 for i in range(len(gt_locations))])
    
    # Create DataFrame
    df = pd.DataFrame({
        'sample_index': range(len(gt_locations)),
        'gt_x': gt_locations[:, 0],
        'gt_y': gt_locations[:, 1],
        'cnn_pred_x': cnn_predictions[:, 0],
        'cnn_pred_y': cnn_predictions[:, 1],
        'cnn_error': cnn_errors,
        'knn_pred_x': nearest_neighbor_predictions[:, 0],
        'knn_pred_y': nearest_neighbor_predictions[:, 1],
        'knn_error': knn_errors,
        'knn_interp_pred_x': nearest_neighbor_interp_predictions[:, 0],
        'knn_interp_pred_y': nearest_neighbor_interp_predictions[:, 1],
        'knn_interp_error': knn_interp_errors
    })
    
    # Save to CSV
    csv_path = os.path.join(output_folder, 'rss_prediction_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"CNN - Mean Error: {np.mean(cnn_errors):.3f} ± {np.std(cnn_errors):.3f} m")
    print(f"K-NN - Mean Error: {np.mean(knn_errors):.3f} ± {np.std(knn_errors):.3f} m")
    print(f"K-NN + Interpolation - Mean Error: {np.mean(knn_interp_errors):.3f} ± {np.std(knn_interp_errors):.3f} m")


def main():
    parser = argparse.ArgumentParser(description='RSS-based Location Prediction Visualizer')
    parser.add_argument('experiment_folder', help='Path to experiment folder containing data-tshark/data.json')
    parser.add_argument('cnn_model', help='Path to CNN model .pth file')
    parser.add_argument('--num-clusters', type=int, default=13,
                        help='Number of clusters for K-means in nearest neighbor algorithm')
    parser.add_argument('--output-folder', default='output',
                        help='Output folder for plots and CSV files')
    
    args = parser.parse_args()
    
    # Find data file
    data_file = os.path.join(args.experiment_folder, 'data-tshark', 'data.json')
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    print(f"Loading RSS data from: {data_file}")
    print(f"Target addresses: {TARGET_ADDRESSES}")
    
    # Load RSS data using existing function
    try:
        inp_rss_vals, gt_locations = loadData_staticTargetAddrMatch(
            data_file, 
            second_hold=5, 
            shuffle=False,
            target_addresses=TARGET_ADDRESSES,
            snap250ms=False  # Following the notebook example
        )
        print(f"Loaded {len(inp_rss_vals)} RSS data points")
    except Exception as e:
        print(f"Error loading RSS data: {e}")
        return
    
    # Load CNN model
    try:
        cnn_model = load_cnn_model(args.cnn_model)
        print(f"Loaded CNN model from: {args.cnn_model}")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return
    
    # Generate predictions
    print("Generating predictions...")
    try:
        (cnn_predictions, nearest_neighbor_predictions, 
         nearest_neighbor_interp_predictions, gt_locations_final) = generate_predictions(
            inp_rss_vals, gt_locations, cnn_model, args.num_clusters
        )
        print("Predictions generated successfully")
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return
    
    # Create plots
    print("Creating plots...")
    create_plots(gt_locations_final, cnn_predictions, nearest_neighbor_predictions, 
                nearest_neighbor_interp_predictions, args.output_folder)
    
    # Save results to CSV
    print("Saving results...")
    save_results_to_csv(gt_locations_final, cnn_predictions, nearest_neighbor_predictions, 
                       nearest_neighbor_interp_predictions, args.output_folder)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 