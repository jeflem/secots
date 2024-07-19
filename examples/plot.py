# This example downloads the boundary of country from OpenStreetMap and finds
# the countries smallest enclosing circle. Results are plotted in 3D (export to
# HTML + JavaScript with Plotly). If the point cloud is not contained in a
# hemisphere, 4 points of the cloud not lying in a hemisphere are highlighted.

# To make this example work you have to install following packages:
# - plotly
# - requests


import numpy as np
import plotly.graph_objects as go
import requests
import secots


# country to show (value of "name:en" key in OSM)
# (for instance, Belgium is contained in a hemisphere, France with its overseas
# territories is not)
COUNTRY = 'Belgium'

# file name for output of interactive plot
HTML_FILE = 'plot.html'

# size of plot
PLOT_WIDTH, PLOT_HEIGHT = 1000, 750

# maximum number of points to plot (affects your webbrowser's performance)
MAX_POINTS = 1000

# gird size in degrees for plotting a sphere
SPHERE_GRID_SIZE = 15

# plot circle as n-gon
CIRCLE_SEGMENTS = 100

# Overpass query string
query = f'''
[output: json][timeout: 600];
rel["type"="boundary"]["admin_level"="2"]["boundary"="administrative"]["name:en"="{COUNTRY}"];
(._; >;);
node._;
out skel;
'''

# send query to Overpass 
print('Sending query to Overpass API...')
r = requests.post(
    'https://overpass-api.de/api/interpreter',
    data={'data': query}
)
assert r.status_code == 200, f'Overpass server returned {r.status_code}.'
objects = r.json()['elements']
assert len(objects) > 0, f'Overpass server doesn\'t return anything.'

# extract points from OSM data
points = [(n['lon'], n['lat']) for n in objects if n.get('type') == 'node']
assert len(points) >= 3, 'There are less than 3 points.'
print(f'Got {len(points)} points from OSM.')

# smallest circle
print('Calculating smallest enclosing circle...')
try:
    lon, lat, r = secots.smallest_circle(points)
    bpoints = None
    print(f'center: ({lon}, {lat}), radius: {r}')
except secots.NotHemisphereError as e:
    bpoints = e.points
    lon, lat, r = None, None, None
    print('Points not contained in hemisphere!')

# prepare plot
fig = go.Figure()
fig.layout.width = PLOT_WIDTH
fig.layout.height = PLOT_HEIGHT

# plot sphere
mesh_lon, mesh_lat = np.meshgrid(
    np.arange(-180, 180, SPHERE_GRID_SIZE),
    np.arange(-90, 90, SPHERE_GRID_SIZE)
)
mesh_lonlat = np.stack((mesh_lon.flatten(), mesh_lat.flatten()), axis=1)
mesh_xyz = secots._lonlat2xyz(mesh_lonlat)
fig.add_trace(
    go.Mesh3d(
        x=mesh_xyz[:, 0], y=mesh_xyz[:, 1], z=mesh_xyz[:, 2],
        color='rgba(0,0,255,0.1)',
        alphahull=0,
        hoverinfo='none'
    )
)

# convert points to xyz
points = secots._lonlat2xyz(np.array(points))
if bpoints is not None:
    bpoints = secots._lonlat2xyz(bpoints)

# reduce number of points for plotting
# (If points are contained in a hemisphere choose 50% with higher probability if
# close to the circle and 50% uniformly at random. This ensures that not too
# many points relevant for circle size are removed and that the cloud's shape is
# still recognizable. If points are not contained in a hemisphere choose points
# uniformly at random.)
if points.shape[0] > MAX_POINTS:
    rng = np.random.default_rng(0)
    if bpoints is None:
        u = secots._lonlat2xyz(np.array([[lon, lat]]))[0, :]  # center
        d = np.cos(r)  # distance of circle's plane to origin
        R = np.sqrt(1 - d ** 2)  # radius of circle in circle's plane
        dists = ((points - d * u.reshape(1, 3)) ** 2).sum(axis=1)  # squared distances
        mask = dists > 0.95 * (R ** 2)
        n = min(mask.sum(), MAX_POINTS // 2)
        points = np.concatenate((
            rng.choice(points[mask], n, replace=False),
            rng.choice(points[np.logical_not(mask)], MAX_POINTS - n, replace=False)
        ), axis=0)
    else:
        points = rng.choice(points, MAX_POINTS, replace=False)

# plot points
fig.add_trace(
    go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        marker={'size': 2, 'color': 'rgba(0,0,255,1)'},
        line={'width': 0, 'color': 'rgba(0,0,0,0)'},
        hoverinfo='none'
    )
)

# plot points not contained in hemisphere
if bpoints is not None:
    fig.add_trace(
        go.Scatter3d(
            x=bpoints[:, 0], y=bpoints[:, 1], z=bpoints[:, 2],
            marker={'size': 2, 'color': 'rgba(255,0,0,1)'},
            line={'width': 0, 'color': 'rgba(0,0,0,0)'},
            hoverinfo='none'
        )
    )

# plot center point
if bpoints is None:
    u = secots._lonlat2xyz(np.array([[lon, lat]]))[0, :]
    fig.add_trace(
        go.Scatter3d(
            x=[u[0]], y=[u[1]], z=[u[2]],
            marker={'size': 4, 'color': 'rgba(255,255,0,1)'},
            line={'width': 0, 'color': 'rgba(0,0,0,0)'},
            hoverinfo='none'
        )
    )

# plot circle
# (The circle is the intersection of a plane and the sphere. We construct the
# plane in parametric form [x, y, z] = p + alpha * v + beta * w with point p
# and directions v and w (unit vectors). Then alpha and beta are chosen to lie
# on a circle of appropriate radius.)
if bpoints is None:
    d = np.cos(r)  # distance of circle's plane to origin
    R = np.sqrt(1 - d ** 2)  # radius of circle in circle's plane
    # fixed point on the plane (starting point for direction vectors)
    p = d * u
    # get one direction vector (orthogonal to u, guaranteed to not be zero)
    v = np.zeros(3)
    i1, i2, i3 = np.argsort(np.abs(u))
    v[i3] = -u[i2]
    v[i2] = u[i3]
    v = v / np.linalg.norm(v)
    # the other direction vector
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    # discretized angles for circle points
    phi = np.linspace(0, 2 * np.pi, CIRCLE_SEGMENTS)
    # reshape all vectors to employ NumPy's broadcasting mechanism
    p = p.reshape(1, 3)
    v = v.reshape(1, 3)
    w = w.reshape(1, 3)
    phi = phi.reshape(-1, 1)
    # circle points
    c = p + R * np.cos(phi) * v + R * np.sin(phi) * w
    # plot circle points
    fig.add_trace(
        go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            marker={'size': 0, 'color': 'rgba(0,0,0,0)'},
            line={'width': 2, 'color': 'rgba(255,255,0,1)'},
            hoverinfo='none'
        )
    )
    
# plot plane
if bpoints is None:
    alpha = np.array([1.1, 1.1, -1.1, -1.1]).reshape(-1, 1)
    beta = np.array([1.1, -1.1, 1.1, -1.1]).reshape(-1, 1)
    c = p + alpha * v + beta * w
    fig.add_trace(
        go.Mesh3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            color='rgba(0,255,0,0.1)',
            hoverinfo='none'
        )
    )

# write HTML with figure
fig.write_html(HTML_FILE)
print(f'HTML file with figure written to {HTML_FILE}.')
