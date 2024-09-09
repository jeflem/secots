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


class _DLLArray:
    '''
    Doubly linked list of fixed size allowing for moving an item to the list's
    top in O(1) time. The list can only hold integers, which are intended to be
    used as indices to some data array.
    '''
    
    def __init__(self, n):
        ''' Create a list of length n. '''
        
        self.first = 0
        self._prevs = [-1] + list(range(0, n - 1))
        self._nexts = list(range(1, n)) + [-1]

    def prev(self, i):
        ''' Get predecessor of item i. Returns -1 if i is the first item. '''
        
        return self._prevs[i]
        
    def next(self, i):
        ''' Get successor of item i. Returns -1 if i is the last item. '''
        
        return self._nexts[i]

    def move_to_top(self, i):
        ''' Move item i to the list's top. '''
        
        if i == self.first:
            return
        
        self._nexts[self._prevs[i]] = self._nexts[i]
        if self._nexts[i] != -1:
            self._prevs[self._nexts[i]] = self._prevs[i]
        self._nexts[i] = self.first
        self._prevs[i] = -1
        self.first = i
    

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


def _welzl(points, bpoints, order, n, hemi_test):
    '''
    Apply a Welzl-type algorithm to find the smallest circle enclosing points
    and having bpoints on its boundary. Raises NotHemisphereError if points are
    not contained in a hemisphere.

    :param ndarray points: NumPy array of shape (n, 3) of points to enclose
                           (xyz coordinates).
    :param ndarray bpoints: NumPy array of shape (m, 3) of boundary points
                           (xyz coordinates).
    :param _DLLArray order: Doubly link list describing in which order points
                            should be processed.
    :param int n: Number of points to process following the given order.
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
        except np.linalg.LinAlgError:  # all 3 points on great circle
            raise NotHemisphereError(_xyz2lonlat(bpoints))
        # more than hemisphere?
        if hemi_test:
            i = order.first
            for _ in range(n):
                if np.dot(points[i, :], u) < t:
                    bpoints_new = np.concatenate((bpoints, [points[i, :]]), axis=0)
                    raise NotHemisphereError(_xyz2lonlat(bpoints_new))
                i = order.next(i)
        return u, t
    
    # make smallest circle for 2 points (including all boundary points)
    if bpoints.shape[0] == 2:
        x1, x2 = bpoints
        skip = 0
    elif bpoints.shape[0] == 1:
        x1 = bpoints[0, :]
        x2 = points[order.first, :]
        skip = 1
    else:
        x1 = points[order.first, :]
        x2 = points[order.next(order.first), :]
        skip = 2
    u = x1 + x2
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-15:  # points antipodal
        bpoints_new = np.concatenate(([x1], [x2]), axis=0)
        raise NotHemisphereError(_xyz2lonlat(bpoints_new))
    u = u / norm_u
    t = (1 + np.dot(x1, x2)) / norm_u

    # check whether points are contained in circle, extend circle if necessary
    i = order.first
    for n_done in range(n):
        if skip > 0:
            skip -= 1
        else:
            dot_prod = np.dot(u, points[i, :])
            if dot_prod < t:
                hemi_test = dot_prod <= -t
                bpoints_new = np.concatenate((bpoints, [points[i, :]]), axis=0)
                u, t = _welzl(points, bpoints_new, order, n_done, hemi_test)
                # move-to-front heuristic
                if order.prev(i) != -1 and order.next(i) != -1:
                    new_i = order.prev(i)
                    order.move_to_top(i)
                    i = new_i
        i = order.next(i)

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

    # remove duplicates
    points = np.unique(points, axis=0)

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
    order = _DLLArray(points.shape[0])
    u, t = _welzl(points, np.empty((0, 3)), order, points.shape[0], hemi_test)
    r = np.arccos(t)
    lon, lat = _xyz2lonlat(u.reshape(1, 3))[0, :]

    return lon, lat, r
