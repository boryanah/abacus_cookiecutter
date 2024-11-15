import numba as nb
import numpy as np

@nb.njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@nb.njit
def wrapping(r1, r2, pos1, pos2, chi1, chi2, Lbox, origin, extra):

    # read dimension of data
    N, nd = pos1.shape
    assert pos1.shape == pos2.shape, "halo positions not of the same dimension"
    assert origin.shape[-1] == nd, "different dimensions of the halo positions and observer's position"
    
    # loop over all halos
    for i in range(N):
        # pos1 fails the condition for selecting halos
        if (r1[i] > chi1) | (r1[i] <= chi2):
            delta = 0.
            for j in range(nd):
                dx = pos1[i][j] - pos2[i][j]
                # halos on opposite sides of the box
                if (dx > Lbox/2.) | (dx <= -Lbox/2.):
                    if pos1[i][j] >= 0.:
                        pos1[i][j] -= Lbox
                    else:
                        pos1[i][j] += Lbox
                dist = pos1[i][j] - origin[j]
                delta += dist*dist
            r1[i] = np.sqrt(delta)
                
        # pos2 fails the condition for selecting halos
        elif (r2[i] > chi1) | (r2[i] <= chi2):
            delta = 0.
            for j in range(nd):
                dx = pos1[i][j] - pos2[i][j]
                # halos on opposite sides of the box
                if (dx > Lbox/2.) | (dx <= -Lbox/2.):
                    if pos2[i][j] >= 0.:
                        pos2[i][j] -= Lbox
                    else:
                        pos2[i][j] += Lbox
                dist = pos2[i][j] - origin[j]
                delta += dist*dist
            r2[i] = np.sqrt(delta)
        # breaks the beau symmetry of the code but this is in cases
        # where main prog and halo are on opposite ends but within r1,2
        # we choose to modify pos2 because for the main box chi_star ~ 1990
        # which is closer to chi1 = 2008 (chi2 = 1865)
        else:
            if np.abs(pos1[i][0] - pos2[i][0]) > extra:
                if pos2[i][0] >= 0:
                    pos2[i][0] -= Lbox
                else:
                    pos2[i][0] += Lbox
                delta = 0.
                for j in range(nd):
                    dist = pos2[i][j] - origin[j]
                    delta += dist*dist
                r2[i] = np.sqrt(delta)
            if np.abs(pos1[i][1] - pos2[i][1]) > extra:
                if pos2[i][1] >= 0:
                    pos2[i][1] -= Lbox
                else:
                    pos2[i][1] += Lbox
                delta = 0.
                for j in range(nd):
                    dist = pos2[i][j] - origin[j]
                    delta += dist*dist
                r2[i] = np.sqrt(delta)
            if np.abs(pos1[i][2] - pos2[i][2]) > extra:
                if pos2[i][2] >= 0:
                    pos2[i][2] -= Lbox
                else:
                    pos2[i][2] += Lbox
                delta = 0.
                for j in range(nd):
                    dist = pos2[i][j] - origin[j]
                    delta += dist*dist
                r2[i] = np.sqrt(delta)

    return r1, r2, pos1, pos2
