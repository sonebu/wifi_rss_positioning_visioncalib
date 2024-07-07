import sys
import math
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta
import threading
import subprocess
import random

import torch, json
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
	
        # Initialize GUI window
        self.setWindowTitle("Indoor Localization Map")
        self.setGeometry(100, 100, 800, 600)

        # Create Matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)

        # Draw your map
        self.draw_map()
        self.device_marker = None

    def draw_map(self):
        # Create a custom axis with a specific range and aspect ratio
        ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])  # Adjust the position and size as needed
        ax.set_xlim(-30, 30)  # Set the x-axis range
        ax.set_ylim(-5, 10)  # Set the y-axis range
        #ax.set_aspect('equal')  # Set the aspect ratio to ensure equal scaling

        # Load map image
        map_image = plt.imread('kroki.jpg')  # Replace 'indoor_map.png' with the path to your map image
        ax.imshow(map_image, extent=[-30, 30, -5, 10])  # Display the map image with the specified extent

        # Add annotations (e.g., markers, labels)
        #self.add_annotation(ax, 'loc', (0, 0), (1, 1), 'red')
        #self.add_annotation(ax, 'Chair', (7, 7), None, 'blue')

        # Set plot parameters
        ax.axis('on')  # Turn off axis ticks and labels
        self.ax = ax
        # Update canvas
        self.canvas.draw()

    def add_annotation(self, ax, text, xy, xytext=None, color='black'):
        ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=20,
                    arrowprops=dict(facecolor=color, arrowstyle='->'))

    def update_device_location(self, location):
        # Clear existing device marker
        if self.device_marker:
        	self.device_marker[0].remove()

        # Plot new device marker
        self.device_marker = self.ax.plot(location[0], location[1], 'bo')  # Blue dot representing the device location
        self.canvas.draw()


def update_location_in_loop(window):
    real_loc=[35,35,35]	
    print("******************loop***********")

    class mdl(nn.Module):
        def __init__(self):
            super(mdl, self).__init__()
            self.input_layer    = nn.Linear(3, 16)
            self.hidden_layer1  = nn.Linear(16, 32)
            self.hidden_layer2  = nn.Linear(32, 20)
            self.output_layer   = nn.Linear(20, 2)
            self.activation_fcn = nn.ReLU()
        def forward(self, x):
            x = self.activation_fcn(self.input_layer(x))
            x = self.activation_fcn(self.hidden_layer1(x))
            x = self.activation_fcn(self.hidden_layer2(x))
            x = self.output_layer(x)
            return x

    # Initialize the model
    model = mdl()

    # Load the state dictionary
    model.load_state_dict(torch.load('model.pth'))

    # Set the model to evaluation mode
    model.eval()

    while True:
        try:
            rssi_data = get_rssi_data()
        except:
            print("rssi info error")
        
            
        
        try:
            rss_1 = rssi_data['d8:84:66:13:53:b8']
            average_rss_1 = sum(rss_1) / len(rss_1)
            real_loc[0]=average_rss_1
        except:
            print('no data 1')
        try:
            rss_2 = rssi_data['d8:84:66:13:53:b0']
            average_rss_2 = sum(rss_2) / len(rss_2)
            real_loc[1]=average_rss_2

        except:
            print('no data 2')
        try:
            rss_3 = rssi_data['d8:84:66:0a:d7:10']
            average_rss_3 = sum(rss_3) / len(rss_3)
            real_loc[2]=average_rss_3
        except:
            print('no data 3')
        print(real_loc)



        device_rss = real_loc
        real_loc = np.asarray(real_loc)
        tensor_real_loc = torch.tensor(real_loc).float()
        estimation = model(tensor_real_loc)
        estimation = estimation.tolist()
        window.update_device_location(estimation)
        print(loc)
        window.canvas.draw()  # Update the canvas to reflect the new device location
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
    time.sleep(3)
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
    
    