'''
Provide the function smallest_circle, which takes a list of longitude-latitude
pairs and returns center and radius of the smallest circle enclosing all points
form the list.

Exception NotHemisphereError is raised if point cloud is not contained in a
hemisphere.
'''


import numpy as np


__all__ = [
    'NotHemisphereError',
    'smallest_circle'
]


class NotHemisphereError(ValueError):
    '''
    Raise if point cloud is not contained in a hemisphere. The points attribute
    contains a (4, 2) shaped NumPy array of 4 points (longtitudes/latitudes)
    from the point cloud identified to be not coverable by a hemisphere.
    '''

    def __init__(self, points):
        self.message = 'Points not contained in a hemisphere!'
        self.points = points
        super().__init__() 


def _lonlat2xyz(lonlat):
    '''
    Convert longitudes/latitudes on the unit sphere to xyz coordinates.
    
    :param ndarray lonlat: NumPy array of shape (n, 2) of longitudes/latitudes.
    :return: NumPy array of shape (n, 3) of xyz coordinates.
    :rtype: ndarray
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
    :return: NumPy array of shape (n, 2) of longitudes/latitudes.
    :rtype: ndarray
    '''

    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(xyz[:, 2])

    return np.stack((lon, lat), axis=1) / np.pi * 180


def _welzl(points, bpoints, hemi_test):
    '''
    Apply a Welzl-type algorithm to find the smallest circle enclosing points
    and having bpoints on its boundary. Raises NotHemisphereError if points are
    not contained in a hemisphere.

    :param ndarray points: NumPy array of shape (n, 3) of points to enclose
                           (xyz coordinates).
    :param ndarray bpoints: NumPy array of shape (m, 3) of boundary points
                           (xyz coordinates).
    :param bool hemi_test: Set to True to check whether points are contained in
                           a hemisphere. If not, NotHemisphereError is raised.
                           Test can be skipped (False) if we are sure that
                           points are contained in a hemisphere.
    :return: Unit vector u (NumPy array of shape (3, )) and number t defining
             the circle as the intersection of the unit sphere and the plane
             np.dot(u, x) == t.
    :rtype: (ndarray, float)
    :raises NotHemisphereError: If point cloud is not contained in a hemisphere.
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
        if hemi_test:
            print('h', end='')
            mask = np.matmul(points, u) < t
            if mask.any():
                new_bpoint = points[np.where(mask)[0][0]]
                bpoints_new = np.concatenate((bpoints, [new_bpoint]), axis=0)
                raise NotHemisphereError(_xyz2lonlat(bpoints_new))
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
        dot_prod = np.dot(u, points[i, :])
        if dot_prod < t:
            hemi_test = dot_prod < -t
            bpoints_new = np.concatenate((bpoints, [points[i, :]]), axis=0)
            u, t = _welzl(points[:i, :], bpoints_new, hemi_test)

    return u, t


def smallest_circle(points, hemi_test=True):
    '''
    Find the smallest circle enclosing all given points on the unit sphere.

    :param ndarray points: NumPy array of shape (n, 2) of points to enclose
                           (longitude/latitude pairs). Compatible types like
                           list of 2-tuples are allowed, too. Compatible is
                           what becomes an (n, 2) array if put into np.array.
    :param bool hemi_test: Set to True (default) to check whether points are
                           contained in a hemisphere. If not, NotHemisphereError
                           is raised. Test can be skipped (False) if you are
                           sure that points are contained in a hemisphere. This
                           saves computation time, but will yield wrong results
                           if points aren't contained in a hemisphere.
    :return: Longitude, latitude of center and radius of smallest enclosing
             circle. Radius is measured along the sphere's surface.
    :rtype: (float, float, float)
    '''

    points = np.array(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Points have to be provided as (n, 2) shaped NumPy array or compatible type!')

    # trivial cases
    if points.shape[0] == 0:
        raise ValueError('Cannot compute smallest enclosing circle for empty set of points!')
    if points.shape[0] == 1:
        return *points[0, :], 0

    # random permutation
    rng = np.random.default_rng(0)
    points = rng.permutation(points, axis=0)

    # convert to xyz
    points = _lonlat2xyz(points)

    # non-trivial case
    u, t = _welzl(points, np.empty((0, 3)), hemi_test)
    r = np.arccos(t)
    lon, lat = _xyz2lonlat(u.reshape(1, 3))[0, :]

    return lon, lat, r
