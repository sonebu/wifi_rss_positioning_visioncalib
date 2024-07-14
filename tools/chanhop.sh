#!/bin/bash
# Channel hopping shell script
# GPLv2
# Portions of code graciously taken from Bill Stearns defragfile
# http://www.stearns.org/defragfile/
#
# jwright@hasborg.com
INTERFACE="mon0"

#BANDS="IEEE80211B IEEE80211G IEEE80211N IEEE80211A"
BANDS="IEEE80211B IEEE80211A"

########################################
#CHANB="1 2 3 4 5 6 7 8 9 10 11"       #               
#CHANBJP="1 2 3 4 5 6 7 8 9 10 11"     #               
#CHANBINTL="1 2 3 4 5 6 7 8 9 10 11"   #              
#CHANA="36 40 44 48 149 153 157 161"   #                
########################################


#########################################################
### To change hopping frequnecies edit below properly ###
CHANB="1 11"                                          ###
CHANBJP=""                                            ###
CHANBINTL=""                                          ###
CHANA="40"                                            ###
#########################################################

echo "Logging hardware details..."
echo "Date: $(date)" 
echo "Interface: $INTERFACE" 
echo "Hardware Details:" 
iwconfig $INTERFACE 
echo "-----------------------------------" 


# Expand specified bands into a list of channels
for BAND in $BANDS ; do
	case "$BAND" in
	IEEE80211B|IEEE80211b|ieee80211b)
		CHANNELS="$CHANNELS $CHANB"
		;;
	IEEE80211BJP|IEEE80211bjp|ieee80211bjp)
		CHANNELS="$CHANNELS $CHANBJP"
		;;
	IEEE80211BINTL|IEEE80211bintl|ieee80211bintl)
		CHANNELS="$CHANNELS $CHANBINTL"
		;;
	IEEE80211A|IEEE80211a|ieee80211a)
		CHANNELS="$CHANNELS $CHANA"
		;;
	*)
		fail "Unsupported band specified \"$BAND\"."
		;;
	esac
done

echo "Starting channel hopping, press CTRL/C to exit."
while true; do
	for CHANNEL in $CHANNELS ; do
		iwconfig $INTERFACE channel $CHANNEL
		if [ $? -ne 0 ] ; then
			fail "iwconfig returned an error when setting channel $CHANNEL"
		fi
        echo "Switched to channel $CHANNEL" 

	done
done