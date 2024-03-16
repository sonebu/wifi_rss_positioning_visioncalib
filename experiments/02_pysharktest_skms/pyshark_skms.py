import pyshark, pickle
cap1 = pyshark.FileCapture("skms_bed_pos1.pcap")

print("loaded capture file")
packetlist_pyshark = []
for packet in cap1:
	packetlist_pyshark.append(packet)

print("extracted packets")
packetlist_dict = []
for pp in packetlist_pyshark:
	packetdict = dict()	
	try:
		packetdict["starttime"] = pp.wlan_radio.start_tsf
		packetdict["endtime"] = pp.wlan_radio.end_tsf
		packetdict["rss"]  = pp.wlan_radio.signal_dbm
		packetdict["freq"] = pp.wlan_radio.frequency
		packetdict["ratembps"] = pp.wlan_radio.data_rate
		packetdict["srcaddr"] = pp.wlan.sa
		packetdict["dstaddr"] = pp.wlan.da
	except:
		continue
	packetlist_dict.append(packetdict)

srcaddrs = list(set([i['srcaddr'] for i in packetlist_dict])) # unique source addresses
#print(len(list(set(srcaddrs))))
print("filtered packets with missing fields")

for srcaddr in srcaddrs:
	srcaddr_filtered = [d for d in packetlist_dict if d['srcaddr'] == srcaddr]
	print(srcaddr)
	for packet in srcaddr_filtered:
		print(packet["rss"])
