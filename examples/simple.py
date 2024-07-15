import numpy as np
import secots

points = np.array([
    [-120, -30],
    [0, -30],
    [120, -30]
])

try:
    lon, lat, r = secots.smallest_circle(points)
    print(f'center: ({lon}, {lat}), radius: {r}')
except secots.NotHemisphereError:
    print('Points not contained in hemisphere!')
