import argparse
import os
import sys
import time
import json
import glob
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from calibration_gui_detectaruco import aruco_detect_draw
import torch


def find_experiment_video(experiment_folder):
    """Find the experiment video file in the given folder."""
    video_patterns = [
        os.path.join(experiment_folder, "experiment_video_*.mp4"),
        os.path.join(experiment_folder, "*.mp4")
    ]
    
    for pattern in video_patterns:
        video_files = glob.glob(pattern)
        if video_files:
            return video_files[0]  # Return the first found video
    
    return None


def validate_experiment_folder(experiment_folder):
    """Validate that the experiment folder contains required files."""
    required_files = [
        "devstring.txt",
        "homography_matrix.npy"
    ]
    
    for file_name in required_files:
        file_path = os.path.join(experiment_folder, file_name)
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_name}")
            return False
    
    # Check devstring.txt format
    devstring_path = os.path.join(experiment_folder, "devstring.txt")
    with open(devstring_path, "r") as f:
        lines = [line.rstrip() for line in f]
    
    if len(lines) != 3:
        print("devstring.txt should have exactly 3 lines")
        return False
    
    return True


def process_video_offline(experiment_folder, video_path, output_suffix="offline"):
    """Process the experiment video offline with YOLO detection."""
    
    print(f"Processing video: {video_path}")
    print(f"Experiment folder: {experiment_folder}")
    
    # Load required files
    homography_matrix = np.load(os.path.join(experiment_folder, "homography_matrix.npy"))
    
    # Initialize YOLO model - exactly like in datacollection_loc.py
    model = YOLO("yolo-Weights/yolov8n.pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Prepare output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_loc_file = os.path.join(experiment_folder, f"loc_xy_{output_suffix}_{timestamp}.txt")
    output_video_file = os.path.join(experiment_folder, f"processed_video_{output_suffix}_{timestamp}.mp4")
    
    # Setup video writer for processed output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    
    # Open location output file
    loc_file = open(output_loc_file, "w")
    
    # Process video frame by frame
    frame_count = 0
    start_time = time.time()
    c = 0  # Counter like in the original
    
    print("Processing frames...")
    
    while True:
        ret, cv_img = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate timestamp based on frame number and FPS
        video_timestamp = frame_count / fps if fps > 0 else frame_count
        
        # run YOLO - exactly like in datacollection_loc.py
        yolo_predictions = model(cv_img, stream=True, verbose=False)
        yolo_detections = np.empty((0, 5))

        # run aruco - exactly like in datacollection_loc.py
        current_time = time.time()
        
        # Print time in the format: HH:MM:SS.ss
        formatted_time = time.strftime("%H:%M:%S", time.localtime(current_time))
        milliseconds = int((current_time * 1000) % 1000)
        ctime = f"{formatted_time}.{milliseconds:03d}"
        centers, corners, ids, cv_img = aruco_detect_draw(cv_img, verbose=False, draw=True)
        plotflag = 0  # Exactly like in the original
        
        for r in yolo_predictions:
            for box in r.boxes:
                classname = int(box.cls[0])  # get detected class name
                if classNames[classname] == "person":  # Exactly like in the original
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                    is_detected_person_arucotagged = False
                    
                    for ii, center in enumerate(centers):
                        is_arucotag_id_correct = True if ids[ii] == 0 else False
                        is_tag_within_box = True if ((center[0] >= x1) and (center[0] <= x2)) and (
                                    (center[1] >= y1) and (center[1] <= y2)) else False
                        if is_arucotag_id_correct and is_tag_within_box:
                            is_detected_person_arucotagged = True
                            plotflag += 1  # Exactly like in the original
                    
                    if is_detected_person_arucotagged:
                        confidence = np.ceil((box.cpu().conf[0] * 100)) / 100  # cast to .cpu()
                        current_detection = np.array([[x1, y1, x2, y2, confidence]])
                        yolo_detections = np.vstack((yolo_detections, current_detection))
                        current_loc = [(x1 + x2) // 2, y2]  # get midpoint at feet
                        current_cord = cv2.perspectiveTransform(np.array([[current_loc]], dtype=np.float32),
                                                                homography_matrix)
                        current_cordx = current_cord[0][0][0]
                        current_cordy = current_cord[0][0][1]
                        timestamp = time.time()
                        timestamp_ordered = datetime.datetime.fromtimestamp(timestamp).strftime(
                            "%H:%M:%S.%f")
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(cv_img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
                        loc_file.write(
                            timestamp_ordered + "," + str(current_cordx) + "," + str(current_cordy) + "\n")
                        c += 1
                    else:
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(cv_img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
                        timestamp = time.time()
                        current_cordx = np.nan
                        current_cordy = np.nan
                        if plotflag == 0:  # Exactly like in the original
                            c += 1
        
        # Write processed frame to output video
        out_video.write(cv_img)
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out_video.release()
    loc_file.close()
    
    processing_time = time.time() - start_time
    print(f"\nProcessing completed!")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processed {frame_count} frames")
    print(f"Output location file: {output_loc_file}")
    print(f"Output video file: {output_video_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Process experiment videos offline with YOLO detection")
    parser.add_argument("-e", "--experiment-folder", required=True,
                        help="Path to the experiment folder containing the video and required files")
    parser.add_argument("-v", "--video-file", required=False,
                        help="Specific video file to process (optional, will auto-detect if not provided)")
    parser.add_argument("-s", "--output-suffix", default="offline",
                        help="Suffix for output files (default: 'offline')")
    
    args = parser.parse_args()
    
    experiment_folder = args.experiment_folder
    
    # Validate experiment folder
    if not os.path.isdir(experiment_folder):
        print(f"Error: Experiment folder does not exist: {experiment_folder}")
        sys.exit(1)
    
    if not validate_experiment_folder(experiment_folder):
        print("Error: Experiment folder validation failed")
        sys.exit(1)
    
    # Find video file
    if args.video_file:
        video_path = args.video_file
        if not os.path.isabs(video_path):
            video_path = os.path.join(experiment_folder, video_path)
    else:
        video_path = find_experiment_video(experiment_folder)
    
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Could not find video file in {experiment_folder}")
        print("Expected patterns: experiment_video_*.mp4 or *.mp4")
        sys.exit(1)
    
    # Process the video
    success = process_video_offline(experiment_folder, video_path, args.output_suffix)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 