# secots - smallest enclosing circles on the sphere

Given a point cloud on a sphere this Python package computes center and radius of the smallest (spherical) circle enclosing the point cloud. The algorithm takes linear time and automatically detects whether the point cloud is contained in a hemisphere. If so, the smalles enclosing circle is returned. Else, an exception is raised.

See [A simple linear time algorithm for smallest enclosing circles on the (hemi)sphere](https://arxiv.org/pdf/2407.19840) for details on the algorithm.

## Installation
Run
```
pip install secots
```
or copy `secots.py` from this repo to your working directory.

## Usage
Put your point cloud into a list of longitude-latitude pairs or into a corresponding (n, 2)-shaped NumPy array. Then call `secots.smallest_circle(points)`. This will return three floats: longitude and latitude of the circle's center as well as the radius. Example:
```python
import secots

points = [
    (-30, 10),
    (0, 20),
    (20, -40),
    (10, 10),
    (20, 30)
]

lon, lat, r = secots.smallest_circle(points)

print(f'center: ({lon}, {lat}), radius: {r}')
```
More examples are in the repo's `example` directory.

The `smallest_circle` function raises `NotHemisphereError` if the point cloud is not contained in a hemisphere. In this case the algorithm is not able to compute a smallest enclosing circle. See [above mentioned paper](https://arxiv.org/pdf/2407.19840) for details.

The `smallest_circle` function accepts an additional keyword argument `hemitest`. Use `hemitest=False` if you are very sure that your point cloud is contained in a hemisphere. This saves (an almost negligible amount of) computation time. Example:
```python
lon, lat, r = secots.smallest_circle(points, hemitest=False)
```

## Contributing
Open a GitiHub issue for bug reports. File pull requests against the `dev` branch.

The `dev` branch contains the code for next release, whereas the `main` branch holds the current release.

## Licence
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html.en)