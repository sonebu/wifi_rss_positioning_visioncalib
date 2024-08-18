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
import torch.optim as optim
import argparse


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


def update_location_in_loop(window,temp_file):
	real_loc = [-35, -35, -35]
	print("******************loop***********")

	#model = torch.load(model_path)
	# Set the model to evaluation mode
	#model.eval()

	while True:
		try:
			rssi_data = get_rssi_data(temp_file)
			
		except:
			print("rssi info error")
		
		try:
			rss_1 = rssi_data['6c:e8:73:d0:4b:76']
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
			
		

def run_tshark_continuously(temp_file):
    # Command to capture detailed RSSI data using tshark continuously
	command = f'tshark -i mon0 -T json -e frame.time -e wlan_radio.signal_dbm -e wlan.sa -e wlan.ssid -e wlan.ra -e wlan.da -e wlan_radio.frequency -e wlan_radio.snr -e wlan_radio.noise_dbm -e wlan.antenna.id -l -j "frame" > {temp_file}'
    
    # Start tshark as a subprocess
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	return process

def get_rssi_data(temp_file):
	rssi_dict = {}
	# Read the JSON file
	try:
		with open(temp_file, 'r+') as file:  # Open in read-write mode
			data = json.load(file)
	except:
		print("load error")
	for line in data:

		try:
			# Each line in the JSON output is a JSON object
			
			layers = line.get('_source', {}).get('layers', {})
			
			# Extract fields from the JSON object, using get with defaults
			time_stamp = layers.get('frame.time', {})
			source_address = layers.get('wlan.sa', {})
			signal_strength = layers.get('wlan_radio.signal_dbm', {})
			ssid = layers.get('wlan.ssid', {})
			receiver_address = layers.get('wlan.ra', {})
			destination_address = layers.get('wlan.da', {})
			frequency = layers.get('wlan_radio.frequency', {})
			snr = layers.get('wlan_radio.snr', {})
			noise_dbm = layers.get('wlan_radio.noise_dbm', {})
			antenna_id = layers.get('wlan.antenna.id', {})


			signal_strength = signal_strength[0]
			source_address = source_address[0]
			# Check if essential fields are present
			if source_address is None or signal_strength is None:
				continue  # Skip this entry if essential fields are missing

			# Convert signal strength to an integer
			signal_strength = int(signal_strength)

			# Store only the source address and RSSI value
			if source_address in rssi_dict:
				# Append the signal strength to the existing list
				rssi_dict[source_address].append(signal_strength)
			else:
				# Initialize a new list with the signal strength
				rssi_dict[source_address] = [signal_strength]
		except json.JSONDecodeError:
			print("Error decoding JSON line")
		except (KeyError, ValueError) as e:
			print(f"Error processing data: {e}")
			continue  # Skip this line

		# Truncate the file to clear its contents
		#file.seek(0)  # Move to the beginning of the file
		#file.truncate(0)  # Clear the contents

	# Print the result for debugging purposes
	#print(rssi_dict)
	return rssi_dict


if __name__ == "__main__":

	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--experiment-folder", required=True, help="Path to the experiment we want to log location data for")
	ap.add_argument("-m", "--model-name", required=False, help="Model to run demo with it")

	args = vars(ap.parse_args())
	folder = os.path.join("experiments", args["experiment_folder"])# Check if the submitted experiment is a valid one
	expcorrect = False
	if not os.path.isdir(folder):
		print("You must provide an existing experiment name")
	else:
		if not os.path.exists(os.path.join(folder, "homography_matrix.txt")):
			print("The experiment does not contain a homography_matrix.txt file (it should)")
		elif not os.path.exists(os.path.join(folder, "reference_image.png")):
			print("The experiment does not contain a reference_image.png file (it should)")
		else:
			# Check if a model name was provided
			if args["model_name"] is not None:
				model_path = os.path.join(folder, args["model_name"])
				if not os.path.exists(model_path):
					print(f"The experiment does not contain a {args['model_name']} file (it should)")
				else:
					# Model file exists
					image_path = os.path.join(folder, 'reference_image.png')
					# Use model_path here if needed
					hmg_mtx_path = os.path.join(folder, 'homography_matrix.txt')
					hmg_mtx = read_hmg_mtx(hmg_mtx_path)
					hmg_mtx_inv = np.linalg.pinv(hmg_mtx)
					expcorrect = True
			else:
				# Model name not provided, continue with the experiment
				image_path = os.path.join(folder, 'reference_image.png')
				hmg_mtx_path = os.path.join(folder, 'homography_matrix.txt')
				hmg_mtx = read_hmg_mtx(hmg_mtx_path)
				hmg_mtx_inv = np.linalg.pinv(hmg_mtx)
				expcorrect = True
				
	if(expcorrect):
		temp_file = 'tshark_temp_output.json'
		# Ensure the temporary file is empty before starting
		if os.path.exists(temp_file):
			os.remove(temp_file)

		tshark_process = run_tshark_continuously(temp_file)

		app = QApplication(sys.argv)
		print("started")
		window = MapWindow()
		window.show()
		print("loop: : : ")
		loop_thread = threading.Thread(target=update_location_in_loop, args=(window,temp_file))
		loop_thread.start()		
		sys.exit(app.exec_())

				



