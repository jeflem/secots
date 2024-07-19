# This example shows how to use secots to find the smallest enclosing circle if
# the point cloud is not known to be contained in a hemisphere.
#
# In case that point cloud is not contained in a hemisphere a message is shown.

import secots

points = [
    (-30, -120),
    (-40, 0),
    (-20, 120),
    (80, 10)
]

try:
    lon, lat, r = secots.smallest_circle(points)
    print(f'center: ({lon}, {lat}), radius: {r}')
except secots.NotHemisphereError:
    print('Points not contained in hemisphere!')
