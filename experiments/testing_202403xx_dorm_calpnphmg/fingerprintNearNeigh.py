import math

def localize(device_rssi):

    known_positions = {
        # 'x_y' corresponds the location of the data point
        # [ap0rta_RSS, apPos_RSS, apNeg_RSS]
        # You can comment out any point from here to use as a test point.
        '0_0': [-34, -63, -52],
        '0_7': [-50, -71, -70],
        '-6_7': [-45, -66, -66],
        '6_7': [-42, -65, -66],
        '-14_2': [-41, -71, -70],
        '14_2': [-48, -53, -53],
        '-28_-4': [-62, -71, -44],
        '-28_4': [-53, -73, -41],
        '28_-4': [-56, -44, -70],
        '28_4': [-53, -41, -72],
        '25_0': [-53, -36, -61],
        '-25_-2': [-54, -63, -37],
        # '-6_4': [-41, -63, -62],
        '7_-2': [-42, -48, -56],
    }

    min_distance = float('inf')
    best_match = None
    for position, fingerprint in known_positions.items():
        distance = math.dist(device_rssi, fingerprint)

        if distance < min_distance:
            min_distance = distance
            best_match = position
    return best_match

device_rss = [-41, -63, -62]
print(localize(device_rss))