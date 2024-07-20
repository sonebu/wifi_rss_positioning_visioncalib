import sys
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import threading
import subprocess
import os
import cv2
import numpy as np

import torch, json
import torch.nn as nn
import numpy as np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
import torch.optim as optim

def read_hmg_mtx(file_path):
    """
    Reads a file containing 9 numbers formatted as a 3x3 matrix and returns it as a numpy array.

    Args:
    file_path (str): Path to the text file.

    Returns:
    np.ndarray: A 3x3 matrix with the numbers from the file.
    """
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            # Read all lines
            lines = file.readlines()
        
        # Combine all lines into a single string and split by whitespace
        numbers = ' '.join(lines).split()
        
        # Convert the list of strings to a list of floats
        numbers = list(map(float, numbers))
        
        # Check if we have exactly 9 numbers
        if len(numbers) != 9:
            raise ValueError("The file does not contain exactly 9 numbers.")
        
        # Reshape the list into a 3x3 numpy array
        matrix = np.array(numbers).reshape(3, 3)
        
        return matrix

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


subfolder = input("Which test do you want to run with ? : ")
#model_name = input("Which model do you want to use ? : ")

nested_subfolder = os.path.join('experiments', subfolder)

image_path = os.path.join(nested_subfolder, 'reference_image.png')
#model_path = os.path.join(nested_subfolder, model_name)

hmg_mtx_path = os.path.join(nested_subfolder, 'homography_matrix.txt')
hmg_mtx = read_hmg_mtx(hmg_mtx_path)
hmg_mtx_inv = np.linalg.pinv(hmg_mtx)



class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize GUI window
        self.setWindowTitle("Indoor Localization Map")
        self.setGeometry(100, 100, 800, 600)

        # Create Matplotlib figure
        # Adjust height ratios and spacing
        self.figure, (self.ax_text, self.ax_map) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 20]})
        self.figure.subplots_adjust(hspace=0.01, top=0.98, bottom=0.02, left=0.05, right=0.95)

        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        '''
        # Create Matplotlib figure
        self.figure, (self.ax_text, self.ax_map) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 20]})
        self.figure.subplots_adjust(hspace=0.01, top=0.95, bottom=0.05, left=0.05, right=0.95)
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        '''

        # Draw your map
        self.draw_map()
        self.device_marker = None

    def draw_map(self):

        # Load map image
        map_image = plt.imread(image_path)  # Replace 'indoor_map.png' with the path to your map image
        self.img_height, self.img_width, _ = map_image.shape

        self.ax_map.imshow(map_image)  # Display the map image with the specified extent

        # Set plot parameters
        self.ax_map.axis('on')  # Turn on axis ticks and labels
        self.ax_text.axis('off')  # Turn off the axis for the text area
        self.canvas.draw()

    def update_device_location(self, location,estimation):
        # Convert the location from pixel coordinates to image coordinates
        x_pixel, y_pixel = location[0], location[1]
        
        # Check if pixel coordinates are within image bounds
        if x_pixel < 0 or x_pixel >= self.img_width or y_pixel < 0 or y_pixel >= self.img_height:
            raise ValueError("Pixel coordinates are out of bounds.")

        # Clear existing device marker
        if self.device_marker:
            self.device_marker[0].remove()
        
        self.device_marker = self.ax_map.plot(x_pixel, y_pixel, 'bo', markersize=10)  
        
        # Clear the text area and add the new estimated location text
        self.ax_text.clear()
        self.ax_text.text(0.5, 0.5, f"Estimated Location: ({estimation[0]:.2f}, {estimation[1]:.2f})",
                          ha='center', va='center', fontsize=40, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
        self.ax_text.axis('off')  # Ensure the text axis stays off

        self.canvas.draw()

def update_location_in_loop(window):
    real_loc = [-35, -35, -35]
    print("******************loop***********")

    #model = torch.load(model_path)
    # Set the model to evaluation mode
    #model.eval()

    while True:
        try:
            rssi_data = get_rssi_data()
            pass
        except:
            print("rssi info error")
        
        try:
            rss_1 = rssi_data['d8:84:66:39:29:e8']
            average_rss_1 = sum(rss_1) / len(rss_1)
            real_loc[0] = average_rss_1
        except:
            print('no data 1')
        try:
            rss_2 = rssi_data['d8:84:66:4a:06:c9']
            average_rss_2 = sum(rss_2) / len(rss_2)
            real_loc[1] = average_rss_2
        except:
            print('no data 2')
        try:
            rss_3 = rssi_data['d8:84:66:03:61:00']
            average_rss_3 = sum(rss_3) / len(rss_3)
            real_loc[2] = average_rss_3
        except:
            print('no data 3')
        print(real_loc)

        device_rss = real_loc
        real_loc = np.asarray(real_loc)
        tensor_real_loc = torch.tensor(real_loc).float()
        #estimation = model(tensor_real_loc)
        #estimation = estimation.tolist()
        for i in range (1,11):
            estimation = [i, i]
            point = np.array(estimation, dtype=np.float32).reshape(1, 1, 2)
            location  = cv2.perspectiveTransform(point, hmg_mtx_inv).reshape(-1).tolist()
            window.update_device_location(location,estimation)
            time.sleep(1)
        
        


def get_rssi_data():
    # Command to capture RSSI data using tshark
    command = 'tshark -i mon0 -T fields -e frame.time -e wlan.sa -e wlan_radio.signal_dbm -e wlan.ssid -a duration:1 | grep "KU"'
    
    # Execute the command and capture the output
    output = subprocess.check_output(command, shell=True, text=True)
    
    # Process the output and extract relevant information
    # Here you'll need to parse the output according to your specific format
    # and extract the relevant RSSI data for location determination
    # You can parse the output using string manipulation, regular expressions, or any other method
    
    # For demonstration, let's assume the output contains RSSI values in a specific format
    print(output)
    # time.sleep(3)
    rssi_dict = {}
    lines = output.strip().split('\n')
    for line in lines:
        fields = line.strip().split('\t')
        if len(fields) == 4:
            time_stamp, source_address, signal_strength, ssid = fields
            if source_address in rssi_dict:
                # If the source address already exists, calculate the average signal strength
                rssi_dict[source_address].append(int(signal_strength))
            else:
                # If the source address doesn't exist, initialize a new list with the signal strength
                rssi_dict[source_address] = [int(signal_strength)]
    print(rssi_dict)
    return rssi_dict



app = QApplication(sys.argv)
print("started")
window = MapWindow()
window.show()
print("loop: : : ")
loop_thread = threading.Thread(target=update_location_in_loop, args=(window,))
loop_thread.start()
	    
sys.exit(app.exec_())
