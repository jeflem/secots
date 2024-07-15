# Provide the function smallest_circle, which takes a list of longitude-latitude
# pairs and returns center and radius of the smallest circle enclosing all
# points form the list.

import numpy as np


__all__ = [
    'NotHemisphereError',
    'smallest_circle'
]


class NotHemisphereError(ValueError):
    ''' Raise if point cloud is not contained in a hemisphere. '''


def _lonlat2xyz(lonlat):
    '''
    Convert longitudes/latitudes on the unit sphere to xyz coordinates.
    
    :param ndarray lonlat: NumPy array of shape (n, 2) of longitudes/latitudes.
    :return ndarray: NumPy array of shape (n, 3) of xyz coordinates.
    '''

    lonlat = lonlat * np.pi / 180

    lon = lonlat[:, 0]
    lat = lonlat[:, 1]

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack((x, y, z), axis=1)


def _xyz2lonlat(xyz):
    '''
    Convert xyz coordinates to longitudes/latitudes on the unit sphere.
    
    :param ndarray xyz: NumPy array of shape (n, 3) of xyz coordinates.
    :return ndarray: NumPy array of shape (n, 2) of longitudes/latitudes.
    '''

    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(xyz[:, 2])

    return np.stack((lon, lat), axis=1) / np.pi * 180


def _welzl(points, bpoints):
    '''
    Apply a Welzl-type algorithm to find the smallest circle enclosing points
    and having bpoints on its boundary. Raises NotHemisphereError if points are
    not contained in a hemisphere.

    :param ndarray points: NumPy array of shape (n, 3) of points to enclose
                           (xyz coordinates).
    :param ndarray bpoints: NumPy array of shape (m, 3) of boundary points
                           (xyz coordinates).
    :return (ndarray, float): Unit vector u (NumPy array of shape (3, )) and
                              number t defining the circle as the intersection
                              of the unit sphere and the plane np.dot(u, x)=t.
    '''

    # 3 boundary points uniquely define the circle
    if bpoints.shape[0] == 3:
        try:
            u = np.linalg.solve(bpoints, np.ones(3))
            norm_u = np.linalg.norm(u)
            u = u / norm_u
            t = 1 / norm_u
        except np.linalg.LinAlgError: # all 3 points on great circle
            x1, x2, x3 = bpoints
            u = np.cross(x2 - x1, x3 - x1)
            u = u / np.linalg.norm(u)
            t = 0
        # more than hemisphere?
        if (np.matmul(points, u) < t).any():
            raise NotHemisphereError()
        return u, t
    
    # make smallest circle for 2 points (including all boundary points)
    n = 2 - bpoints.shape[0]  # number of non-boundary points to consider
    x1, x2 = np.concatenate((points[:n, :], bpoints), axis=0)
    u = x1 + x2
    norm_u = np.linalg.norm(u)
    u = u / norm_u
    t = (1 + np.dot(x1, x2)) / norm_u

    # check whether points are contained in circle
    for i in range(2 - bpoints.shape[0], points.shape[0]):
        if np.dot(u, points[i, :]) < t:
            bpoints_new = np.concatenate((bpoints, [points[i, :]]), axis=0)
            u, t = _welzl(points[:i, :], bpoints_new)

    return u, t


def smallest_circle(points):
    '''
    Find the smallest circle enclosing all given points on the unit sphere.
    Raises NotHemisphereError if points are not contained in a hemisphere.

    :param ndarray points: NumPy array of shape (n, 2) of points to enclose
                           (longitude/latitude pairs).
    :return (float, float, float): Longitude, latitude of center and radius of
                                   smallest enclosing circle. Radius is measured
                                   along the sphere's surface.
    '''

    # trivial cases
    if points.shape[0] == 0:
        raise ValueError('Cannot compute smallest enclosing circle for empty set of points!')
    if points.shape[0] == 1:
        return *points[0, :], 0

    # convert to xyz
    points = _lonlat2xyz(points)

    # non-trivial case
    u, t = _welzl(points, np.empty((0, 3)))
    r = np.arccos(t)
    lon, lat = _xyz2lonlat(u.reshape(1, 3))[0, :]

    return lon, lat, r
