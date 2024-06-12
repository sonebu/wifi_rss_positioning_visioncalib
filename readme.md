# WiFi based Indoor Self-Positioning Experiments

Benchmarking and developing no-reference RSS-based indoor positioning systems (i.e., input = only RSS, output = location predictions). We use a calibrated camera-based setup to collect ground truth location information and simultaneously sample input RSS data points over wireshark for training and evaluation. Demos do not require any live input from the camera (only the reference image is used for visualization). 

--- 
## Description

4 stages are considered in an experiment:

**1- Calibration:** Set up camera and calibrate it for the given scene (currently manual, can be automated) to obtain the homography matrix for a reference image.

**2- Data Collection:** Run a person detection and tracking model and push the outputs through the homography matrix to obtain the actual 2D positions of the person with timestamps. Simultaneously record the RSS values with timestamps. Merge the two sets of data points and obtain a single set of labeled {loc_x, loc_y, RSS values} data points.

**3- Development:** For neural network based algorithms this is training and evaluation, for readily-available algorithms or non-parametric ones (e.g., nearest neighbour (NeNe), NeNe+interp etc.) skip this part. Evaluation can use the whole dataset collected or it can be done with data points not used in training depending on what type of algorithm is used. 

**4- Demo:** Run the algorithm with new inputs, and a GUI shows realtime updates overlaid on the reference image taken during the calibration step. **Note that running the demo anywhere other than the room it was calibrated in does not make sense since the updates overlaid on the reference image will be invalid.**

Therefore, each experiment has a "data package" associated with it. The data package contains:

- a reference image from a calibrated and stationary camera ("refimg.png")
- RSS + camera-based loc_x/loc_y data collected for the room seen in that reference image ("data.json")

Different algorithms can be developed for and demonstrated on that data package.

![System Diagram](docs/drawings/system.png)


---

## Installation and Usage

#### For Development
...

#### For Demonstrations
...


