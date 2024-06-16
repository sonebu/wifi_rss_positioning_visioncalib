# turn off wifi and disconnect all networks first
nmcli radio wifi off # turns wifi interface off + gives an rfkill signal
rfkill unblock wlan  # turns of the rfkill signal without changing wifi interface state
sudo iw phy phy0 interface add mon0 type monitor # adds monitor mode interface to the network build
sudo ip link set mon0 up # turns the monitor mode interface on

# wireshark should be ready to go at this point
