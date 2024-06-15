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

def localize(device_rssi):
    known_positions = {
        # 'x_y' corresponds the location of the data point
        # [ap0rta_RSS, apPos_RSS, apNeg_RSS]
        # You can comment out any point from here to use as a test point.
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
        '-6_4': [-41, -63, -62],
        '7_-2': [-42, -48, -56],
    }

    min_distance = float('inf')
    best_match = None
    for position, fingerprint in known_positions.items():
        distance = math.dist(device_rssi, fingerprint)

        if distance < min_distance:
            min_distance = distance
            best_match = position
    return best_match
def update_location_in_loop(window):

    while True:
        rssi_data = get_rssi_data()
        try:
            
            rss_1 = rssi_data['24:81:3b:2d:bf:80']
            rss_2 = rssi_data['24:81:3b:4e:fe:a0']
            rss_3 = rssi_data['24:81:3b:2d:aa:e0']
            
            #random_number_1= random.randint(-100, -30)
            #random_number_2 = random.randint(-100, -30)
            #random_number_3 = random.randint(-100, -30)
            
            average_rss_1 = sum(rss_1) / len(rss_1)
            average_rss_2 = sum(rss_2) / len(rss_2)
            average_rss_3 = sum(rss_3) / len(rss_3)

            device_rss = [average_rss_1, average_rss_2, average_rss_3]
            #device_rss = [random_number_1,random_number_2,random_number_3]
            
            loc = localize(device_rss)
            loc = loc.split("_")
            loc = [eval(i) for i in loc]
            window.update_device_location(loc)
            print(loc)
            window.canvas.draw()  # Update the canvas to reflect the new device location
            time.sleep(1)
        except Exception as e:
            print("Error:", e)



def get_rssi_data():
    # Command to capture RSSI data using tshark
    command = 'tshark -i mon0 -T fields -e frame.time -e wlan.sa -e wlan_radio.signal_dbm -e wlan.ssid -a duration:2 | grep "KU"'
    
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
    return rssi_dict

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    loop_thread = threading.Thread(target=update_location_in_loop, args=(window,))
    loop_thread.start()
	    
    sys.exit(app.exec_())
    
    
