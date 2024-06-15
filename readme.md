# WiFi based Indoor Self-Positioning Experiments

Benchmarking and developing no-reference RSS-based indoor positioning systems (i.e., input = only RSS, output = location predictions). We use a calibrated camera-based setup to collect ground truth location information and simultaneously sample input RSS data points over wireshark for training and evaluation. Demos do not require any live input from the camera (only the reference image is used for visualization). 

--- 
## Description

4 stages are considered in an experiment:

**1- Calibration:** Set up a fixed camera, get a reference image + a homography matrix over known actual points in that image (currently manual, will probably be automated with ArUco tags).

**2- Data Collection:** Run a person tracking model to obtain the actual 2D positions of the person (w.r.t. ground w/ homography) with timestamps. Simultaneously record the RSS values with timestamps. Merge these to get {loc_x, loc_y, RSS values}. Landmark detection for feet with a pose model could work too.

**3- Development:** Training and evaluating neural net based algorithms, dictionary selection for non parametric algorithms (e.g., nearest neighbour (NeNe), NeNe+interp etc.). 

**4- Demo:** Run the algorithm with new inputs, and a GUI shows realtime updates overlaid on the reference image taken during the calibration step. **Running the demo anywhere other than the room it was calibrated in does not make sense since the updates overlaid on the reference image will be invalid.**

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


