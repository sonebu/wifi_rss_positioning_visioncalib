List of tools:

- calibration_gui.py: GUI used for shooting the ref image as well as finding the homography matrix
- datacollection_loc.py: script that takes in experiment folderpath, reads devstring, shows video, runs YOLO and writes timestamped loc preds to file
- datacollection_rss.py: script that takes in experiment folderpath, dumps timestamped tshark readings to file 
- datacollection_merge.py: script that takes in experiment folderpath, merges \_loc.py and \_rss.py outputs to create data.json

the other files are auxiliary.

datacollection_loc_sort.py is a direct replica of https://github.com/abewley/sort/blob/master/sort.py
