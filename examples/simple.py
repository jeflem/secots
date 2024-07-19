# This example shows how to use secots to find the smallest enclosing circle if
# the point cloud is known to be contained in a hemisphere.
#
# Note that we do not have to know the hemisphere.

import secots

points = [
    (-30, 10),
    (0, 20),
    (20, -40),
    (10, 10),
    (20, 30)
]

lon, lat, r = secots.smallest_circle(points, hemi_test=False)

print(f'center: ({lon}, {lat}), radius: {r}')
