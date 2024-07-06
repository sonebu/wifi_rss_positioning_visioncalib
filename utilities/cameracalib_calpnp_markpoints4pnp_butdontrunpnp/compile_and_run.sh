#!/bin/bash

# Compile the C++ code
cd /home/kadir/ELEC491/calpnp
g++ -I/usr/local/include/opencv4 -L/usr/local/lib/ -g -o bin ./src/main.cpp ./src/Cfg.cpp ./src/CamCal.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -lm

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the binary..."

    # Execute the binary
    ./bin

    cd /home/kadir/ELEC491
    echo "Running Python script..."
    python3 cv.py




else
    echo "Compilation failed."
fi
