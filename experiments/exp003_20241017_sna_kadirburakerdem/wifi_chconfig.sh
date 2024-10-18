#!/bin/bash
# Channel hopping shell script
# GPLv2
# Portions of code graciously taken from Bill Stearns defragfile
# http://www.stearns.org/defragfile/
#
# jwright@hasborg.com

# Defaults
BANDS="IEEE80211B"
DWELLTIME=".50"

CHANB="1" # 2 7 3 8 4 9 5 10""
CHANBJP="" #"1 13 6 11 2 12 7 3 8 14 4 9 5 10"
CHANBINTL="" # "1 13 6 11 2 12 7 3 8 4 9 5 10"
CHANA="" # "36 40 44 48 52 56 60 149 153 157 161"

INTERFACE="mon0"

	
iwconfig $INTERFACE channel $CHANB


