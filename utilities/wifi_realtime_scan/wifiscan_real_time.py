from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import json
from datetime import datetime
from collections import deque


BUFFER_SIZE = 100



def frametime2secs(frametime):
    hrs   = int(frametime.split(":")[0][-2:])
    mins  = int(frametime.split(":")[1])
    secs  = int((frametime.split(":")[2]).split(".")[0])
    msecs = int((frametime.split(":")[2]).split(".")[1][:2])
    total_seconds = (hrs*60*60*1000 + mins*60*1000 + secs*1000 + msecs*10) / 1000.0 
    return total_seconds

ofset = frametime2secs(str(datetime.now()))

def get_timeinsecs(sampledict, offset=0):
    return frametime2secs(sampledict["_source"]["layers"]["frame.time"][0]) - offset;

# Tshark command
command = "tshark -i mon0 -T json -e frame.time -e wlan_radio.signal_dbm -e wlan.sa -e wlan.ssid -e wlan.ra -e wlan.da -e wlan_radio.frequency -e wlan_radio.snr -e wlan_radio.noise_dbm -e wlan.antenna.id"


plt.ion()
# Setup the figure
fig, ax = plt.subplots(figsize=(16, 5), dpi=80)
line, = ax.plot([], [], marker="*", linestyle="None")
ax.set_ylim([-95, 0])
ax.grid()
plt.show()


def filterbySSID(time):
    ####### Filter data by unique source addresses and get lists
    lo_uniqueSAs = []
    lo_freqs = []
    lo_SActrs = []
    lo_SSIDs = []

    for pkt in packets:

        if(pkt["_source"]["layers"]["wlan.sa"] not in lo_uniqueSAs):
            lo_uniqueSAs.append(pkt["_source"]["layers"]["wlan.sa"])
            lo_freqs.append(pkt["_source"]["layers"]["wlan_radio.frequency"][0])
            lo_SActrs.append(1)
            lo_SSIDs.append(pkt["_source"]["layers"]["wlan.ssid"][0])
        else:
            position_of_uniqueSA_in_listofuniqueSAs = lo_uniqueSAs.index(pkt["_source"]["layers"]["wlan.sa"])
            lo_SActrs[position_of_uniqueSA_in_listofuniqueSAs] += 1

    # Print results
    print("SSID | Source Address    |  Freq |  # of occurrences")
    print("-" * 50)
    KU_AP_idx = []

    for i, ssid in enumerate(lo_SSIDs):
        print(f"{ssid}  | {lo_uniqueSAs[i]} | {lo_freqs[i]} | {lo_SActrs[i]}")
        KU_AP_idx.append(i)


    ############ Plot the RSS values for each Source        


    def get_freq(sampledict):
        return sampledict["_source"]["layers"]["wlan_radio.frequency"][0]

    def get_sourceaddr(sampledict):
        return sampledict["_source"]["layers"]["wlan.sa"]
        
    def get_ssid(sampledict):
        return sampledict["_source"]["layers"]["wlan.ssid"][0]
        
    def get_rss(sampledict):
        return sampledict["_source"]["layers"]["wlan_radio.signal_dbm"][0]

    
    array_ssid_rss = np.zeros((len(time), len(KU_AP_idx)))*np.nan
    for sample in packets:
        t_sample  = get_timeinsecs(sample, offset=ofset)
        t_idx     = time.index(t_sample)
        sa_sample = get_sourceaddr(sample)
        sa_idx    = lo_uniqueSAs.index(sa_sample)
        if(sa_idx in KU_AP_idx):
            array_ssid_idx = KU_AP_idx.index(sa_idx)
            array_ssid_rss[t_idx, array_ssid_idx] = get_rss(sample)

    # Update the plot
    ax.clear()
    ax.set_ylim([-95, 0])
    ax.grid()
    ax.plot(time, array_ssid_rss, marker="*")
    ax.legend([(lo_uniqueSAs[idx][0] + ", count: " + str(lo_SActrs[idx]) + ", freq: " + str(lo_freqs[idx])) for idx in KU_AP_idx], 
              prop=font_manager.FontProperties(family='monospace'), loc='upper center', ncol=3)
    plt.draw()
    plt.pause(0.01)

def run(command):
    process = Popen(command, stdout=PIPE, shell=True)
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        yield line.decode('utf-8')


if __name__ == "__main__":
    packets = deque(maxlen=BUFFER_SIZE)
    current_packet = ""

    for path in run(command):
        if not path == "[":
            current_packet += path.strip()
            if path.strip() == "},":
                item = json.loads(current_packet[:-1])
                frame_time = item.get("_source", {}).get("layers", {}).get("frame.time", [""])[0]
                wlan_ssid = item.get("_source", {}).get("layers", {}).get("wlan.ssid", [""])[0]
                if wlan_ssid != "KU":
                    current_packet = ""
                    continue
                transformed_item = {
                    "_index": item.get("_index", "unknown"),
                    "_type": item.get("_type", "doc"),
                    "_score": item.get("_score", None),
                    "_source": {
                        "layers": {
                            "frame.time": [frame_time],
                            "wlan_radio.signal_dbm": item.get("_source", {}).get("layers", {}).get("wlan_radio.signal_dbm", []),
                            "wlan.sa": item.get("_source", {}).get("layers", {}).get("wlan.sa", []),
                            "wlan.ra": item.get("_source", {}).get("layers", {}).get("wlan.ra", []),
                            "wlan.da": item.get("_source", {}).get("layers", {}).get("wlan.da", []),
                            "wlan_radio.frequency": item.get("_source", {}).get("layers", {}).get("wlan_radio.frequency", []),
                            "wlan.ssid": [wlan_ssid]
                        }
                    }
                }    
                packets.append(transformed_item)
                time = []
                
                
                #To make plot with time indez commentout the below sentence.
                #packets = [sample for sample in packets if current_time - get_timeinsecs(sample) <=15]
                
                for sample in packets:
                    t = get_timeinsecs(sample, offset=ofset)
                    if(t not in time):
                        time.append(t)


                
                current_packet = ""

                current_time = frametime2secs(str(datetime.now()))
                print("packets length : ", len(packets))
                print("Current time : ", current_time - ofset);
                print("Last packet time: ", time[-1])

                filterbySSID(time)



                
           

         
        
