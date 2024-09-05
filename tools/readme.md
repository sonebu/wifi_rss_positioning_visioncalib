List of tools:

- calibration_gui.py: GUI used for shooting the ref image as well as finding the homography matrix
- datacollection_loc.py: script that takes in experiment folderpath, reads devstring, shows video, runs YOLO and writes timestamped loc preds to file
- datacollection_rss.sh: script that takes in experiment folderpath, dumps timestamped tshark readings to file and write output into rss_value.txt

- datacollection_merge.py: script that takes in experiment folderpath, merges \_loc.py and \_rss.py outputs to create data.json (kullanılmıyor)

- datacollection_merge.sh: Script that takes an experiment folder path, runs the output of datacollection_loc.py and datacollection_rss.sh, merges the data, and converts the merged data to JSON format. The output file is output.json. (input flagi sizin yaptığınız formatla aynı yaptım) (./datacollection_merge.sh -e ../../experiments/exp000_202403xx_dorm_calpnphmg)

the other files are auxiliary.

datacollection_loc_sort.py is a direct replica of https://github.com/abewley/sort/blob/master/sort.py

wifi_chconfig_template.sh: Simple Bash script that changes the channel to monitor different frequencies.







