# turn off wifi and disconnect all networks first
sudo iw phy phy0 interface add mon0 type monitor
sudo ip link set mon0 up

# this needs work