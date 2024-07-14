#!/bin/bash

INTERFACE="mon0"

tshark -i $INTERFACE -T fields -e frame.time -e wlan_radio.signal_dbm -e wlan.sa -e wlan.ssid  -l > rss_value.txt
