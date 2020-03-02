#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:38:02 2019

@author: brsr
"""

import geopandas
import shapely
from shapely.geometry import LineString, Polygon, Point
import pyproj
import homography
import warnings
import numpy as np
from abc import ABC
from scipy.optimize import minimize, minimize_scalar, root_scalar

#TODO:
#implement conformal (some kind of circle-packing thing?)
#script comparisons

#arange3 = np.arange(3)
#FIRST AXIS IS SPATIAL

TGTPTS3 = np.eye(3)
TGTPTS4 = np.array([[0, 1, 1, 0],
                    [0, 0, 1, 1]])

def sqrt(x):
    """Real sqrt clipped to 0 for negative values.

    >>> x = np.array([-np.inf, -1, 0, 1, np.inf, np.nan])
    >>> sqrt(x)
    array([ 0.,  0.,  0.,  1., inf, nan])
    """
    return np.where(x < 0, 0, np.sqrt(x))

def geodesics(lon, lat, geod, n=100):
    """Draw geodesics between each adjacent pair of points given by
    lon and lat.
    """
    lon2 = np.roll(lon, -1, axis=0)
    lat2 = np.roll(lat, -1, axis=0)
    result = []
    for l, t, l2, t2 in zip(lon, lat, lon2, lat2):
        g = geod.npts(l, t, l2, t2, n)
        g.insert(0, (l, t))
        g.append((l2, t2))
        result.append(LineString(g))
    return geopandas.GeoSeries(result)

def ptseriestoarray(ser):
    """Convert a geopandas GeoSeries containing shapely Points
    (or LineStrings of all the same length) to an array of
    shape (2, n) or (3, n).
    """
    return np.stack([x.coords for x in ser], axis=-1).squeeze()

def arraytoptseries(arr):
    """Convert an array of shape (2, ...) or (3, ...) to a
    geopandas GeoSeries containing shapely Point objects.
    """
    if arr.shape[0] == 2:
        return geopandas.GeoSeries([Point(x[0], x[1])
                                    for x in arr.reshape(2, -1).T])
    else:
        return geopandas.GeoSeries([Point(x[0], x[1], x[2])
                                    for x in arr.reshape(3, -1).T])

def fixbary_normalize(bary):
    """Converts array bary to an array with sum = 1 by dividing by
    bary.sum(). Will return nan if bary.sum() == 0.

    >>> fixbary_normalize(np.arange(3))
    array([0.        , 0.33333333, 0.66666667])
    """
    bary = np.array(bary)
    return bary / np.sum(bary, axis=0, keepdims=True)

def fixbary_subtract(bary):
    """Converts array bary to an array with sum = 1 by subtracting
    (bary.sum() - 1)/bary.shape[0].

    >>> fixbary_subtract(np.arange(3))
    array([-0.66666667,  0.33333333,  1.33333333])
    """
    bary = np.array(bary)
    s = (np.sum(bary, axis=0, keepdims=True) - 1)/bary.shape[0]
    return bary - s

def transeach(func, geoms):
    """Transform each element of geoms using the function func."""
    plist = []
    for geom in geoms:
        plist.append(shapely.ops.transform(func, geom))
    return geopandas.GeoSeries(plist)

def lltovector(lon, lat, scale=np.pi/180):
    """Convert longitude and latitude to 3-vector

    >>> ll = np.arange(6).reshape(2,3)*18
    >>> lltovector(ll[0],ll[1])
    array([[5.87785252e-01, 2.93892626e-01, 4.95380036e-17],
           [0.00000000e+00, 9.54915028e-02, 3.59914664e-17],
           [8.09016994e-01, 9.51056516e-01, 1.00000000e+00]])
    """
    x = np.cos(lat*scale)*np.cos(lon*scale)
    y = np.cos(lat*scale)*np.sin(lon*scale)
    z = np.sin(lat*scale)

    return np.stack([x, y, z], axis=0)

def vectortoll(vector, scale=180/np.pi):
    """Convert 3-vector to longitude and latitude

    >>> vectortoll(np.eye(3))
    array([[ 0., 90.,  0.],
           [ 0.,  0., 90.]])
    """
    #vector = vector/np.linalg.norm(vector, axis=0, keepdims=True)
    lat = scale*np.arctan2(vector[2], np.sqrt(vector[1]**2 + vector[0]**2))
    lon = scale*np.arctan2(vector[1], vector[0])
    return np.stack([lon, lat])

def graticule(spacing1=15, spacing2=1,
              lonrange = [-180, 180], latrange = [-90, 90]):
    """
    Create a graticule (or another square grid)
    """
    a = (lonrange[1] - lonrange[0])/spacing2
    b = (latrange[1] - latrange[0])/spacing1
    c = (lonrange[1] - lonrange[0])/spacing1
    d = (latrange[1] - latrange[0])/spacing2
    plx = np.linspace(lonrange[0], lonrange[1], num=a + 1)
    ply = np.linspace(latrange[0], latrange[1], num=b + 1)
    mex = np.linspace(lonrange[0], lonrange[1], num=c + 1)
    mey = np.linspace(latrange[0], latrange[1], num=d + 1)
    parallels = np.stack(np.meshgrid(plx, ply), axis=-1).transpose((1,0,2))
    meridians = np.stack(np.meshgrid(mex, mey), axis=-1)
    gratlist = [parallels[:, i] for i in range(parallels.shape[1])]
    gratlist += [meridians[:, i] for i in range(meridians.shape[1])]
    gratl2 = [LineString(line) for line in gratlist]
    grat = geopandas.GeoSeries(gratl2)
    grat.crs = {'init': 'epsg:4326'}
    return grat

def barygrid(spacing1=0.1, spacing2=1E-2, rang = [0, 1], eps=1E-8):
    """Create a barycentric grid
    """
    nx = int((rang[1] - rang[0])/spacing1 + 1)
    ny = int((rang[1] - rang[0])/spacing2 + 1)
    x = np.linspace(rang[0], rang[1], nx)
    y = np.linspace(rang[0], rang[1], ny)
    z = 1 - x[..., np.newaxis] - y
    #valid = (rang[0] <= z) & (z <= rang[1])
    #z[~valid] = np.nan
    bary1 = np.stack([np.broadcast_to(x[..., np.newaxis], (nx, ny)),
              np.broadcast_to(y, (nx, ny)),
              z])
    bary = np.concatenate([bary1, np.roll(bary1, -1, axis=0),
                     np.roll(bary1, -2, axis=0)], axis=1)
    gratlist = [bary[:, i] for i in range(nx*3)]
    gratl2 = []
    for i in range(nx*3):
        g = gratlist[i]
        valid = np.all((rang[0]-eps <= g) & (g <= rang[1]+eps), axis=0)
        if np.sum(valid) > 1:
            g = g[..., valid]
            gratl2.append(LineString(g.T))
    grat = geopandas.GeoSeries(gratl2)
    grat.crs = {'init': 'epsg:4326'}
    return grat
#%%
def trigivenangles(angles, scale=np.pi/180):
    """Given angles, create the vertices of a triangle with those vertex
    angles. Only uses the first 2 angles. The last vertex is always 1, 0.
    >>> angles = np.array([45,90,45])
    >>> np.round(trigivenangles(angles), decimals=8)
    array([[-1.,  0.,  1.],
           [ 0., -1.,  0.]])
    """
    angles = angles * scale
    p0 = [np.cos(2*angles[1]), np.sin(2*angles[1])]
    p1 = [np.cos(2*angles[0]), np.sin(-2*angles[0])]
    p2 = [1, 0]
    return np.array([p0, p1, p2]).T

def trigivenlengths(sidelengths):
    """Given side lengths, creates the vertices of a triangle with those
    side lengths, and having circumcenter at 0,0.
    >>> sidelengths=np.array( [3,4,5])
    >>> np.round(trigivenlengths(sidelengths), decimals=8)
    array([[-2.5, -0.7,  2.5],
           [ 0. , -2.4,  0. ]])
    """
    #might be more stable to use law of cotangents, but eh
    r = np.product(sidelengths)/sqrt(
            2*np.sum(sidelengths**2*np.roll(sidelengths,1)**2)
            -np.sum(sidelengths**4))
    s1 = sidelengths
    s2 = np.roll(sidelengths, -1)
    s3 = np.roll(sidelengths, 1)
    cosangle = (s2**2 + s3**2 - s1**2)/ (2*s2*s3)
    angles = np.arccos(cosangle)
    return r*trigivenangles(np.roll(angles, -1), scale=1)
#%%
# def authalic_lat(phi, geod, scale = np.pi/180):
#     """Convert geodetic latitude to authalic
#     """
#     phi = phi * scale
#     e   = np.sqrt(geod.es)
#     qp  = 1 + (1 - e**2)/e * np.arctanh(e)
#     q   = ((1 - e**2)*np.sin(phi)/(1 - (e*np.sin(phi))**2) +
#            (1 - e**2)/e * np.arctanh(e*np.sin(phi)) )
#     return np.arcsin(q/qp)/scale

# def gerp(v1, v2, t, geod):
#     """Geodetic interpolation"""
#     f, b, r = geod.inv(v1[0], v1[1], v2[0], v2[1])
#     lon, lat, b = geod.fwd(v1[0], v1[1], f, r*t)
#     return lon, lat

# def invgerp(v1, v2, vt, geod):
#     """Inverse geodetic interpolation"""
#     f, b, r = geod.inv(v1[0], v1[1], v2[0], v2[1])
#     f, b, rt = geod.inv(v1[0], v1[1], vt[0], vt[1])
#     return rt/r

# def gintersect(a0, a1, b0, b1, geod, initial=None):
#     """Find point of intersection of geodesic from a0 to a1 and
#     from b0 to b1
#     """
#     def objective(t):
#         pt = gerp(a0, a1, t, geod=geod)
#         area, _ = geod.polygon_area_perimeter([b0[0], b1[0], pt[0]],
#                                               [b0[1], b1[1], pt[1]])
#         return area
#     if initial is None:
#         x0 = 1
#     else:
#         x0 = invgerp(a0, a1, initial, geod)
#     x1 = 1 if abs(x0) < 0.5 else 0
#     res = root_scalar(objective, x0=x0, x1=x1)
#     if not res.converged:
#         print(res)
#     return gerp(a0, a1, res.root, geod=geod)

def central_angle(x, y, signed=False):
    """Central angle between vectors with respect to 0. If vectors have norm
    1, this is the spherical distance between them.

    Args:
        x, y: Coordinates of points on the sphere.
        axis: Which axis the vectors lie along. By default, -1.

    Returns: Array of central angles.

    >>> t = np.linspace(0, np.pi, 5)
    >>> c = np.cos(t)
    >>> s = np.sin(t)
    >>> z = np.zeros(t.shape)
    >>> x = np.stack((c, s, z), axis=-1)
    >>> y = np.stack((c, z, s), axis=-1)
    >>> np.round(central_angle(x, y)/np.pi*180)
    array([  0.,  60.,  90.,  60.,   0.])
    """
    cos = np.sum(x*y, axis=-1)
    sin = np.linalg.norm(np.cross(x, y), axis=-1)
    result = np.arctan2(sin, cos)
    return result if signed else abs(result)

def slerp(pt1, pt2, intervals):
    """Spherical linear interpolation.

    Args:
        pt1: Array of points. When interval is 0, the result is pt1.
        pt2: Array of points. When interval is 1, the result is pt2.
        intervals: Array of intervals at which to evaluate the
            linear interpolation

    >>> x = np.array([1, 0, 0])
    >>> y = np.array([0, 0, 1])
    >>> t = np.linspace(0, 1, 4)[:, np.newaxis]
    >>> slerp(x, y, t)
    array([[1.       , 0.       , 0.       ],
           [0.8660254, 0.       , 0.5      ],
           [0.5      , 0.       , 0.8660254],
           [0.       , 0.       , 1.       ]])
    """
    t = intervals
    angle = central_angle(pt1, pt2)[..., np.newaxis]
    return (np.sin((1 - t)*angle)*pt1 + np.sin((t)*angle)*pt2)/np.sin(angle)

def shoelace(pts):
    """Find area of polygon in the plane defined by pts, where pts is an
    array with shape (2,n).
    >>> pts = np.arange(6).reshape(2,-1)%4
    >>> shoelace(pts)
    2.0
    """
    return abs(np.sum(np.cross(pts, np.roll(pts, -1, axis=1), axis=0)))/2

def omegascale(adegpts, degpts_t, actrlpts, tgtpts, geod):
    """
    """
    ar, p = geod.polygon_area_perimeter(actrlpts[0], actrlpts[1])
    #at = shoelace(tgtpts)
    es = geod.es
    a = geod.a
    factor = np.pi/180
    #lon = adegpts[0]*factor
    lat = adegpts[1]*factor
    x = degpts_t[0]
    y = degpts_t[1]
    dx = np.gradient(x, factor, edge_order=2)
    dy = np.gradient(y, factor, edge_order=2)
    dxdlat, dxdlon = dx
    dydlat, dydlon = dy
    J = (dydlat*dxdlon - dxdlat*dydlon)
    R = a*np.sqrt(1-es)/(1-es*np.sin(lat)**2)
    h = sqrt((dxdlat)**2 + (dydlat)**2)*(1-es*np.sin(lat)**2)**(3/2)/(a*(1-es))
    k = sqrt((dxdlon)**2 + (dydlon)**2)*(1-es*np.sin(lat)**2)**(1/2)/(a*np.cos(lat))
    scale = J/(R**2*np.cos(lat))
    sinthetaprime = np.clip((scale/(h*k)), -1, 1)
    aprime = sqrt(h**2 + k**2 + 2*h*k*sinthetaprime)
    bprime = sqrt(h**2 + k**2 - 2*h*k*sinthetaprime)
    sinomegav2 = np.clip(bprime/aprime, -1, 1)
    omega = 360*np.arcsin(sinomegav2)/np.pi
    #scale = h*k*sinthetaprime#*at/ar
    return omega, scale

#%%
class Projection(ABC):
    """Don't subclass this without subclassing one of
    transform and transform_v and one of invtransform and invtransform_v,
    or else an infinite regression will occur"""
    def transform(self, x, y, z = None, **kwargs):
        if z is None:
            pts = np.stack([x,y])
        else:
            pts = np.stack([x,y,z])
        vresult = self.transform_v(pts, **kwargs)
        return vresult

    def invtransform(self, x, y, z=None, **kwargs):
        if z is None:
            pts = np.stack([x,y])
        else:
            pts = np.stack([x,y,z])
        vresult = self.invtransform_v(pts, **kwargs)
        return vresult

    def transform_v(self, pts, **kwargs):
        rpts = pts.reshape((pts.shape[0],-1)).T
        result = []
        for xy in rpts:
            result.append(self.transform(*xy, **kwargs))
        result = np.array(result)
        shape = [result.shape[-1], ] + list(pts.shape[1:])
        return result.T.reshape(shape)

    def invtransform_v(self, pts, **kwargs):
        rpts = pts.reshape((pts.shape[0],-1)).T
        result = []
        for xy in rpts:
            result.append(self.invtransform(*xy, **kwargs))
        result = np.array(result)
        shape = [result.shape[-1], ] + list(pts.shape[1:])
        return result.T.reshape(shape)
#%%
class Bilinear(Projection):
    """Bilinear interpolation
    """
    _bilinear_mat = np.array([[ 1, 1, 1, 1],
                              [-1, 1, 1,-1],
                              [-1,-1, 1, 1],
                              [ 1,-1, 1,-1]])/4
    def __init__(self, tgtpts):
        self.tgtpts = tgtpts
        self.abcd = self._bilinear_mat @ tgtpts.T

    def transform(self, u, v):
        """u and v should have the same shape"""
        abcd = self.abcd
        stack = np.stack([np.ones(u.shape), u, v, u*v])
        return (abcd @ stack).T

    def transform_v(self, pts, **kwargs):
        return self.transform(pts[0], pts[1])

    def invtransform_v(self, pts):
        abcd = self.abcd
        A = abcd[:,0]
        B = abcd[:,1]
        C = abcd[:,2]
        D = abcd[:,3] - pts
        AB = np.cross(A,B)
        AC = np.cross(A,C)
        AD = np.cross(A,D)
        BC = np.cross(B,C)
        BD = np.cross(B,D)
        CD = np.cross(C,D)
        ua = 2*BD
        ub = AD + BC
        uc = 2*AC
        va = 2*CD
        vb = AD - BC
        vc = 2*AB
        u1 = (-ub + sqrt(ub**2 - ua*uc) )/ua
        #u2 = (-ub - sqrt(ub**2 - ua*uc) )/ua
        #v2 = (-vb + sqrt(vb**2 - va*vc) )/va
        v1 = (-vb - sqrt(vb**2 - va*vc) )/va
        return u1, v1

class Homeomorphism(Projection):
    """Homeomorphism"""
    def __init__(self, tgtpts):
        self.tgtpts = tgtpts

class Barycentric(Projection):
    """Transforms between plane and barycentric coordinates"""
    def __init__(self, tgtpts):
        self.tgtpts = tgtpts
        m = np.concatenate([self.tgtpts, np.ones((1, 3))])
        self.minv = np.linalg.inv(m)

    def transform_v(self, bary):
        """Convert barycentric to plane"""
        rbary = bary.reshape(3,-1)
        result = self.tgtpts @ rbary
        shape = [2,] + list(bary.shape[1:])
        return result.reshape(shape)

    def invtransform_v(self, xy):
        """Convert plane to barycentric"""
        rxy = xy.reshape(2,-1)
        shape = list(rxy.shape)
        shape[0] = 1
        xy1 = np.concatenate([rxy, np.ones(shape)])
        result = self.minv @ xy1
        shape = [3,] + list(xy.shape[1:])
        return result.reshape(shape)

class MapProjection(Projection, ABC):
    def __init__(self, ctrlpts, geod):
        """Parameters:
        ctrlpts: 2x3 or 2x4 Numpy array, latitude and longitude of 
            each control point
        geod= a pyproj.Geod object. For a unit sphere use
            pyproj.Geod(a=1,b=1)
        """
        self.geod = geod
        self.ctrlpts = ctrlpts
        self.ctrlvector = lltovector(*ctrlpts)
        faz, baz, sidelengths = self.geod.inv(ctrlpts[0], ctrlpts[1],
                                          np.roll(ctrlpts[0], -1),
                                          np.roll(ctrlpts[1], -1))
        area, _ = geod.polygon_area_perimeter(*ctrlpts)
        self.sidelengths = sidelengths
        self.faz = faz
        self.baz = baz
        self.ctrlarea = area
        antipode = ctrlpts.copy()
        antipode[0] -= 180
        index = antipode[0] < -180
        antipode[0, index] += 360
        antipode[1] *= -1
        self.ctrlantipode = antipode
        self.ctrlantipodevector = lltovector(*antipode)
        if ctrlpts.shape[1] == 4:
            ctrlvector = self.ctrlvector 
            v0 = ctrlvector[..., 0]
            v1 = ctrlvector[..., 1]
            v2 = ctrlvector[..., 2]
            v3 = ctrlvector[..., 3]
            poip1 = np.cross(np.cross(v0,v1), np.cross(v3,v2))
            poip2 = np.cross(np.cross(v0,v3), np.cross(v1,v2))
            poip = np.stack([[poip1,-poip1],[poip2,-poip2]]).transpose(2,0,1)
            poip = poip / np.linalg.norm(poip, axis=0)
            self.poiv = poip
            self.poi = vectortoll(poip)        

    def orienttgtpts(self, tgtpts, N = (0, 90)):
        """Orient target points so that line from 0 to the projection of N
        points up. Will fail if map projection doesn't define tgtpts."""
        pN = self.transform(*N)
        if np.allclose(pN, [0,0]):
            raise ValueError('projection of N too close to 0')
        angle = np.arctan2(pN[0],pN[1])
        rotm = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
        result = rotm @ tgtpts
        self.tgtpts = result

#%%
class ChambTrimetric(MapProjection):
    """Chamberlin trimetric projection"""
    def __init__(self, ctrlpts, geod):
        super().__init__(ctrlpts, geod)
        self.tgtpts = trigivenlengths(self.sidelengths)
        try:
            self.orienttgtpts(self.tgtpts)
        except ValueError:
            pass

    def transform(self, x, y, **kwargs):
        if hasattr(x, '__iter__'):
            raise TypeError()
        tgtpts = self.tgtpts
        f, b, rad = self.geod.inv(self.ctrlpts[0], self.ctrlpts[1],
                                   x*np.ones(3), y*np.ones(3))
        faz = self.faz
        raz1 = (faz - f) % 360
        radsq = np.array(rad).squeeze()**2
        ctgt = tgtpts.T.copy().view(dtype=complex).squeeze()
        a = np.roll(ctgt, -1) - ctgt
        b = ctgt
        l = abs(a)
        lsq = l**2
        rsq = radsq/lsq
        ssq = np.roll(radsq, -1, axis=-1)/lsq
        x0 = (rsq - ssq + 1)/2
        y0 = sqrt(-rsq**2 + 2*rsq*(ssq + 1) - (ssq - 1)**2)/2
        y0[np.isnan(y0)] = 0
        y = np.where(raz1 > 180, -y0, y0)
        z0 = x0 +1j*y
        pts = (a * z0 + b)
        result = np.mean(pts)
        return result.real, result.imag

    def invtransform(self, *args, **kwargs):
        return NotImplemented
#%%
class LstSqTrimetric(ChambTrimetric):
    """Least-squares variation of the Chamberlin trimetric projection"""
    def transform(self, x, y, **kwargs):
        init = super().transform(x, y)
        tgtpts = self.tgtpts
        f, b, rad = self.geod.inv(self.ctrlpts[0], self.ctrlpts[1],
                                   x*np.ones(3), y*np.ones(3))
        def objective(v):
            x = v[0]
            y = v[1]
            a = tgtpts[0]
            b = tgtpts[1]
            xma = x-a
            ymb = y-b
            dist = np.sqrt(xma**2 + ymb**2)
            result = np.sum((dist - rad)**2 )
            f = 1 - rad/dist
            f[rad <= 0] = 1
            jac = 2*np.array([np.sum(xma*f), np.sum(ymb*f)])
            return result, jac
        res = minimize(objective, init, jac=True,
                       method = 'BFGS')
        return res.x
#%%
class LinearTrimetric(MapProjection):
    """The linear variation of the Chamberlin Trimetric projection."""
    matrix1 = np.array([[0,-1],
               [1,0]])
    matrix2 = np.array([[0, -1, 1],
               [1, 0, -1],
               [-1, 1, 0]])
    matrixinv1 = np.array([[-2,1,1],
              [1,-2,1],
              [1,1,-2]])*2/3

    def __init__(self, ctrlpts, geod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        vctrl = self.ctrlvector
        self.radius = ((geod.a**(3/2) + geod.b**(3/2))/2)**(2/3)
        self.invctrlvector = np.linalg.pinv(vctrl)
        self.invperpmatrix = np.linalg.pinv(vctrl.T @ vctrl)
        self.tgtpts = trigivenlengths(self.sidelengths)
        self.setmat()
        try:
            self.orienttgtpts(self.tgtpts)
            self.setmat()
        except ValueError:
            pass

    def setmat(self, tgtpts=None):
        """Set matrices that use tgtpts"""
        if tgtpts is None:
            tgtpts = self.tgtpts
        else:
            self.tgtpts = tgtpts
        tgtde = np.linalg.det(np.concatenate([tgtpts, np.ones((1,3))], axis=0))
        self.m = self.matrix1 @ tgtpts @ self.matrix2 /(2*tgtde)
        self.minv = self.matrixinv1 @ tgtpts.T

    def transform_v(self, pts):
        rpts = pts.reshape((2,-1)).T
        rad = []
        for x,y in rpts:
            f, b, radi = self.geod.inv(x*np.ones(3), y*np.ones(3),
                          self.ctrlpts[0], self.ctrlpts[1])
            rad.append(radi)
        shape = list(pts.shape)
        shape[0] = 3
        rad = np.array(rad).T
        radsq = np.array(rad)**2
        result = self.m @ radsq
        return result.reshape(pts.shape)

    def invtransform_v(self, pts, n=20, stop=1E-8):
        if not self.geod.sphere:
            warnings.warn('inverse transform is approximate on ellipsoids')
        rpts = pts.reshape((2,-1))
        k = self.minv @ rpts/self.radius**2
        hmin = -np.min(k, axis=0)
        #hmax = np.pi**2-np.max(k, axis=0)
        h = hmin.copy()
        for i in range(n):
            rsq = (k + h)
            #pos = rsq > 0
            neg = rsq < 0
            zer = rsq == 0
            c = np.where(neg, np.cosh(np.sqrt(-rsq)), np.cos(np.sqrt(rsq)))
            b = np.where(neg, np.sinh(np.sqrt(-rsq)),
                         np.sin(np.sqrt(rsq)))/np.sqrt(np.abs(rsq))
            b[zer] = 1
            f = np.einsum('i...,ij,j...', c, self.invperpmatrix, c) - 1
            fprime = np.einsum('i...,ij,j...', c, self.invperpmatrix, b)
            delta = f/fprime
            h += delta
            if np.max(np.abs(delta)) < stop:
                break
        #h = np.clip(h, hmin, hmax)
        rsq = np.clip(k + h, 0, np.pi**2)
        c = np.cos(np.sqrt(rsq))
        vector = self.invctrlvector.T @ c
        return vectortoll(vector).reshape(pts.shape)

    def nmforplot(self, pts, n=100):
        rpts = pts.reshape((2,-1))
        k = self.minv @ rpts/self.radius**2
        hmin = -np.min(k, axis=0)
        hmax = np.pi**2-np.max(k, axis=0)
        h = np.linspace(hmin,hmax,100).T
        rsq = (k[..., np.newaxis] + h)
        c = np.cos(np.sqrt(rsq))
        nm = np.einsum('i...,ij,j...', c, self.invperpmatrix, c)
        return h, nm
#%%
class Areal(MapProjection):
    """Spherical areal projection."""
    def __init__(self, ctrlpts, geod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod: a pyproj.Geod object. For a unit sphere use
                pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        #it's possible to get a geod where this would give the wrong answer,
        #but it's not likely
        area, _ = geod.polygon_area_perimeter([0,120,-120],[0,0,0])
        self.totalarea = 2*area
        self.ctrlarea, _ = geod.polygon_area_perimeter(ctrlpts[0],
                                                        ctrlpts[1])
        vctrl = lltovector(ctrlpts[0], ctrlpts[1])
        self.ctrlvector = vctrl
        a_i = np.sum(np.roll(self.ctrlvector, -1, axis=1) *
                          np.roll(self.ctrlvector, 1, axis=1), axis=0)
        self.a_i = a_i
        self.b_i = (np.roll(a_i, -1) + np.roll(a_i, 1))/(1+a_i)
        self.tau_c = self.tau(self.ctrlarea)

    def tau(self, area):
        """Convert areas on the geod to tau values for inverse transform"""
        return np.tan(area/self.totalarea*2*np.pi)

    def transform(self, x, y):
        try:
            areas = []
            for i in range(3):
                smtri = self.ctrlpts.copy()
                smtri[:, i] = np.array([x,y])
                a, _ = self.geod.polygon_area_perimeter(smtri[0],
                                                         smtri[1])
                areas.append(a)
            areas = np.array(areas)
            return areas/self.ctrlarea
        except ValueError:
            raise TypeError()

    def invtransform_v(self, bary):
        rbary = bary.reshape(3,-1)
        if not self.geod.sphere:
            warnings.warn('inverse transform is approximate on ellipsoids')
        b_i = self.b_i[:,np.newaxis]
        tau = self.tau_c
        tau_i = self.tau(self.ctrlarea*rbary)
        t_i = tau_i/tau
        c_i = t_i / ((1+b_i) + (1-b_i) * t_i)
        f_i = c_i / (1 - np.sum(c_i, axis=0))
        vector = self.ctrlvector @ f_i
        shape = [2] + list(bary.shape[1:])
        result = vectortoll(vector).reshape(shape)
        return result
#%%
class SphereProjection(MapProjection):
    geod = pyproj.Geod(a=1, b=1)

    def __init__(self, ctrlpts, k=0):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts, self.geod)
        self.k = k

    def transform(self, *args, **kwargs):
        return NotImplemented

    def _fix_corners(self, *args, **kwargs):
        if self.ctrlpts.shape[1] == 4:
            return self._fix_corners_uv(*args, **kwargs)
        elif self.ctrlpts.shape[1] == 3:
            return self._fix_corners_bary(*args, **kwargs)

    def _fix_corners_uv(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        index3 = (lon == ctrlpts[0,3]) & (lat == ctrlpts[1,3])
        result[..., index0] = np.array([-1, -1])[..., np.newaxis]
        result[..., index1] = np.array([ 1, -1])[..., np.newaxis]
        result[..., index2] = np.array([ 1,  1])[..., np.newaxis]
        result[..., index3] = np.array([-1,  1])[..., np.newaxis]
        return result

    def _fix_corners_bary(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        result[..., index0] = np.array([1, 0, 0])[..., np.newaxis]
        result[..., index1] = np.array([0, 1, 0])[..., np.newaxis]
        result[..., index2] = np.array([0, 0, 1])[..., np.newaxis]
        return result

    def _fix_corners_inv(self, *args, **kwargs):
        if self.ctrlpts.shape[1] == 4:
            return self._fix_corners_inv_uv(*args, **kwargs)
        elif self.ctrlpts.shape[1] == 3:
            return self._fix_corners_inv_bary(*args, **kwargs)

    def _fix_corners_inv_uv(self, x, y, result):
        index0 = (x == -1) & (y == -1)
        index1 = (x == 1) & (y == -1)
        index2 = (x == 1) & (y == 1)
        index3 = (x == -1) & (y == 1)
        result[..., index0] = self.ctrlvector[..., 0, np.newaxis]
        result[..., index1] = self.ctrlvector[..., 1, np.newaxis]
        result[..., index2] = self.ctrlvector[..., 2, np.newaxis]
        result[..., index3] = self.ctrlvector[..., 3, np.newaxis]
        return result

    def _fix_corners_inv_bary(self, bary, result):
        index0 = (bary[0] == 1)
        index1 = (bary[1] == 1)
        index2 = (bary[2] == 1)
        if np.any(index0):
            result[..., index0] = self.ctrlvector[..., 0, np.newaxis]
        if np.any(index1):
            result[..., index1] = self.ctrlvector[..., 1, np.newaxis]
        if np.any(index2):
            result[..., index2] = self.ctrlvector[..., 2, np.newaxis]
        return result
#%%
class BisectTri(SphereProjection):
    def transform(self, lon, lat):
        vtestpt = lltovector(lon, lat)
        areas = []
        vctrlpts = self.ctrlvector
        actrlpts = self.ctrlpts
        geod = self.geod
        area = self.ctrlarea
        for i in range(3):
            vc = np.roll(vctrlpts, i, axis=1)
            ac = np.roll(actrlpts, i, axis=1)
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            lllproj = vectortoll(lproj)
            a1, _ = geod.polygon_area_perimeter(list(ac[0]) + [lllproj[0]],
                                                list(ac[1]) + [lllproj[1]])
            areas.append(a1)
        areas = np.array(areas)
        aa = areas/area
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, i, axis=0)
            b = (y**2 * x**2 + z**2 * x**2 - y**2 * z**2
                - x * y**2 + z * y**2
                - 2*y*x**2 - x*z**2 + y*z**2 + x**2
                + 3*y*x + z*x - 2*y*z
                - 2*x - y + z + 1)
            bx.append(b)
        bx = np.array(bx)
        betax = bx/bx.sum()
        return self._fix_corners(lon, lat, betax)

    def invtransform(self, b1, b2, b3):
        return NotImplemented

#%%
class FullerTri(SphereProjection):
    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.tweak = tweak

    def fixbary(self, bary):
        if self.tweak:
            return fixbary_normalize(bary)
        else:
            return fixbary_subtract(bary)

    def transform(self, lon, lat):
        vtestpt = lltovector(lon, lat)
        ctrlvector = self.ctrlvector
        b = []
        for i in range(3):
            v0 = ctrlvector[..., i]
            v1 = ctrlvector[..., (i+1)%3]
            v2 = ctrlvector[..., (i+2)%3]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt20 = vtestpt @ np.cross(v2, v0)        
            cosw01 = v0 @ v1
            cosw02 = v0 @ v2
            w01 = np.arccos(cosw01)
            w02 = np.arccos(cosw02)
            #print(w01, w02)            
            if np.isclose(w01, w02):
                #print('equal')
                w = (w01 + w02) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                g = vt12 + cosw*(vt01 + vt20)
                ti = self._b_eq(w, sinw, vt20, vt01, g)
            else:
                #print('notequal')
                sinw01 = sqrt(1 - cosw01**2)
                sinw02 = sqrt(1 - cosw02**2)
                g = vt12 + cosw02*vt01 + cosw01*vt20                        
                ti = self._b_neq(w01, sinw02, vt01, w02, sinw01, vt20, g)
            b.append(1-ti)
        return self.fixbary(b)
    
    def _b_neq(self, w01, sinw02, vt01, w02, sinw01, vt20, g):
        t0 = (w01*sinw02*vt01 + w02*sinw01*vt20)/(g*w01*w02)
        if ~np.isfinite(t0):
            t0 = 0
        else:
            lim = np.pi/np.array([w01,w02]).max()
            t0 = np.clip(t0, -lim, lim)
            
        if abs(t0) < 1E-3:
            return t0
        w = (w01 + w02) / 2
        sinw = np.sin(w)
        t1 = self._b_eq(w, sinw, vt20, vt01, g)
        t0 = np.clip(t0, -abs(t1), abs(t1))
        c1 = sqrt(g**2 + (sinw01*vt20 - sinw02*vt01)**2)
        c2 = sqrt(g**2 + (sinw01*vt20 + sinw02*vt01)**2)
        d1 = np.arctan2(sinw01*vt20 - sinw02*vt01, g)
        d2 = np.arctan2(sinw01*vt20 + sinw02*vt01, g)        
        def objective(t):
            if t < -lim or t > lim:
                return t**2, 2*t
            if t == 0:
                t = np.finfo(float).eps
            z = c1*np.cos((w01 - w02)*t - d1) - c2*np.cos((w01 + w02)*t - d2)
            dz = (-c1*(w01 - w02)*np.sin((w01 - w02)*t - d1) 
                  + c2*(w01 + w02)*np.sin((w01 + w02)*t - d2))
            return z/t, (t*dz - z)*t**-2        
        res = root_scalar(objective, fprime=True, method='newton', x0=t0)
        return res.root
        
    def _b_eq(self, w, sinw, vt20, vt01, gx):
        #x = sinw*(vt20 + vt01)/gx
        tx = np.arctan2(sinw*(vt20 + vt01),gx)/w                
        #this form would be more efficient:
        #b = np.arctan2(sinw*vt12, cosw*vt12 + vt01 + vt20)/w
        return tx
        
    def invtransform(self, b1, b2, b3):
        return NotImplemented
#%%
class FullerQuad(SphereProjection):
    
    def transform(self, lon, lat):
        vtestpt = lltovector(lon, lat)
        ctrlvector = self.ctrlvector
        result = []
        for p in [(1,3),(3,1)]:
            #print(p)
            v0 = ctrlvector[..., 0]
            v1 = ctrlvector[..., p[0]]
            v2 = ctrlvector[..., 2]
            v3 = ctrlvector[..., p[1]]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt23 = vtestpt @ np.cross(v2, v3)
            vt30 = vtestpt @ np.cross(v3, v0)
            
            vt02 = vtestpt @ np.cross(v0, v2)
            vt13 = vtestpt @ np.cross(v1, v3)
            cosw01 = v0 @ v1
            #cosw12 = v1 @ v2
            cosw23 = v2 @ v3
            #cosw30 = v3 @ v0
            
            w01 = np.arccos(cosw01)
            #w12 = np.arccos(cosw12)
            w23 = np.arccos(cosw23)
            #w30 = np.arccos(cosw30)
            sinw01 = sqrt(1 - cosw01**2)
            #sinw12 = sqrt(1 - cosw12**2)
            sinw23 = sqrt(1 - cosw23**2)
            #sinw30 = sqrt(1 - cosw30**2)     
            if np.isclose(w01, w23):
                w = (w01 + w23) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                j = self._b_eq(w, sinw, cosw, vt01, vt02, 
                               vt12, vt13, vt23, vt30)
            else:
                #print('ne')
                j = self._b_neq(w01, w23, vt12, vt30, vt02, vt13, 
                                cosw01, sinw01, cosw23, sinw23)
            result.append(j)
        return result
    
    def _b_neq(self, w01, w23, vt12, vt30, vt02, vt13, 
               cosw01, sinw01, cosw23, sinw23):
        rx2 = [vt12 - vt30 * (cosw01 * cosw23 + sinw01 * sinw23)
                   - cosw01 * vt02 - cosw23 * vt13, 
               vt30 * (sinw23 * cosw01 - sinw01 * cosw23 )
                   + sinw23 * vt13 - sinw01 * vt02,
                   + cosw01 * vt02 + cosw23 * vt13 -vt12 
                   + vt30 * (cosw01 * cosw23 - sinw01 * sinw23),
               vt30 * (sinw01 * cosw23 + sinw23 * cosw01 )
                   + sinw01 * vt02 + sinw23 * vt13]
        c1 = sqrt(rx2[0]**2 +rx2[1]**2)
        c2 = sqrt(rx2[2]**2 +rx2[3]**2)
        d1 = np.arctan2(rx2[1], rx2[0])
        d2 = np.arctan2(rx2[3], rx2[2])
        wm = w01 - w23
        wp = w01 + w23
        #theta01 = np.arccos(v0 @ poip)
        #theta23 = np.arccos(v3 @ poip)
        #slerp(v0, v1, -thetap/w01) = slerp(v3, v2, -thetap2/w23) = poip
        #lims = [theta01/w01, theta23/w23
        #lim = 2
        def objective(t):
            #if t < -lim or t > lim:
            #    return t**2, 2*t
            z = c1*np.cos(wm*t - d1) + c2*np.cos(wp*t - d2)
            dz = -c1*wm*np.sin(wm*t - d1) - c2*wp*np.sin(wp*t - d2)
            return z, dz
        
        res = root_scalar(objective, fprime=True, method='newton', x0=0.5)
        return res.root
        
    def _b_eq(self, w, sinw, cosw, vt01, vt02, vt12, vt13, vt23, vt30):
        #print(w, sinw, cosw, vt01, vt02, vt12, vt13, vt23, vt30)
        a = vt12 - cosw * (vt02 + vt13) - vt30 * cosw**2 
        b = sinw * (2 * vt30 * cosw + vt02 + vt13)
        c = -vt30 * sinw**2

        #print(a,b,c)
        if a == 0:
            num = -c
            denom = b     
        else:
            desc = b**2 - 4*a*c
            if b > 0:#FIXME buhhhh
                num = -b + sqrt(desc)
            else:
                num = -b - sqrt(desc)
            denom= 2*a                 
        j = np.arctan2(num,denom)/w
        #print(j)
        return j

    def invtransform(self, u, v):
        a = self.ctrlvector[..., 0]
        b = self.ctrlvector[..., 1]
        c = self.ctrlvector[..., 2]
        d = self.ctrlvector[..., 3]
        f = slerp(a,b,u)
        g = slerp(d,c,u)
        h = slerp(b,c,v)
        k = slerp(a,d,v)
        inv = np.cross(np.cross(f, g), np.cross(h, k))
        return vectortoll(inv)
#%%
class SnyderEA(SphereProjection):
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlvector = self.ctrlvector
        v_0 = ctrlvector[..., 0]
        v_1 = ctrlvector[..., 1]
        v_2 = ctrlvector[..., 2]
        self.v_01 = v_0 @ v_1
        self.v_12 = v_1 @ v_2
        self.v_20 = v_2 @ v_0
        self.v_012 = np.linalg.det(ctrlvector)
        self.c = self.v_12
        self.c2 = self.c**2
        self.s2 = 1 - self.c2
        self.s = sqrt(self.s2)
        self.w = np.arccos(self.c)

    def transform(self, lon, lat):
        actrlpts = self.ctrlpts
        ctrlvector = self.ctrlvector
        area = self.ctrlarea
        geod = self.geod
        vtestpt = lltovector(lon, lat)
        lproj = -np.cross(np.cross(ctrlvector[..., 1], ctrlvector[..., 2]),
                          np.cross(ctrlvector[..., 0], vtestpt))
        norm = np.linalg.norm(lproj, axis=0, keepdims=True)
        if norm != 0:
            lproj = lproj / norm
        lllproj = vectortoll(lproj)
        cosAP = ctrlvector[..., 0] @ vtestpt
        cosAD = ctrlvector[..., 0] @ lproj
        pl = sqrt((1-cosAP)/(1-cosAD))
        b0 = 1 - pl
        lona = list(actrlpts[0,:2]) + [lllproj[0],]
        lata = list(actrlpts[1,:2]) + [lllproj[1],]
        a1, _ = geod.polygon_area_perimeter(lona, lata)
        b2 = a1/area * pl
        b1 = 1 - b0 - b2
        #if any([np.isnan(b0), np.isnan(b1), np.isnan(b2)]):
        #    print(lon, lat, b0, b1, b2)
        result = np.stack([b0,b1,b2])
        bresult = self._fix_corners_bary(lon, lat, result)
        return np.where(np.isfinite(bresult), bresult, 0)

    def invtransform(self, b1, b2, b3):
        ctrlvector = self.ctrlvector
        #actrlpts = self.ctrlpts
        area = self.ctrlarea
        #geod = self.geod
        lp = np.array(1-b1)
        #make this an array so it won't complain about zero division, impute later
        a = b3/lp
        v_01 = self.v_01
        v_20 = self.v_20
        v_012 = self.v_012
        c = self.c
        s = self.s
        w = self.w
        Ar = a * area
        sA = np.sin(Ar)
        cA = 1 - np.cos(Ar)
        Fp = ((sA * v_012 + cA*(v_01*c - v_20))**2 - (s*cA*(1 + v_01))**2)
        Gp = 2*cA*s*(1 + v_01)*(sA*v_012 + cA*(v_01*c - v_20))
        #x = Gp/Fp
        result = 1/w*np.arctan2(Gp, Fp)
        vd = slerp(ctrlvector[..., 1], ctrlvector[..., 2], result)
        AdotD = ctrlvector[..., 0] @ vd
        AdotP = 1 - lp**2*(1-AdotD)
        t = np.arccos(AdotP)/np.arccos(AdotD)
        vresult = slerp(ctrlvector[..., 0], vd, t)
        bary = np.stack([b1, b2, b3])
        vresult = self._fix_corners_inv_bary(bary, vresult)
        vresult = np.where(np.isfinite(vresult),vresult,0)
        return vectortoll(vresult)

class SnyderEASym(SphereProjection):
    """this doesn't really accomplish anything"""
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        subproj = []
        for i in range(3):
            cp = np.roll(ctrlpts, i, axis=1)
            pj = SnyderEA(cp)
            subproj.append(pj)
        self.subproj = subproj
            
    def transform(self, lon, lat):
        subproj = self.subproj
        for i in range(3):
            pj = subproj[i]
            b = np.roll(pj.transform(lon, lat), -i, axis=0)
            #print(b)
            try:
                beta += b
            except NameError:
                beta = b
        return beta/3
    
#%%
class NSlerpTri(SphereProjection):
    def _tri_naive_slerp_angles(self, bary, pow=1, eps=0):
        """Interpolates the angle factor so that it's equal to the
        angle between pts 1 and 2 when beta_3=0, etc.
        """
        angles = self.sidelengths
        if (np.max(angles) - np.min(angles)) <= eps:
            return np.mean(angles)
        #if np.allclose(angles):
        #    return angles[..., 0]
        a = bary[0]
        b = bary[1]
        c = bary[2]
        ab = (a*b)**pow
        bc = (b*c)**pow
        ca = (c*a)**pow
        denom = ab + bc + ca
        numer = ab*angles[0] + bc*angles[1] + ca*angles[2]
        return numer/denom

    def invtransform_v(self, bary, pow=1):
        base = self.ctrlvector
        angles = self._tri_naive_slerp_angles(bary, pow)
        b = np.sin(angles * bary) / np.sin(angles)
        result = (b.T.dot(base.T)).T
        result = self._fix_corners_inv(bary, result)
        return vectortoll(result)

class NSlerpQuad(SphereProjection):
    #FIXME convert to [0,1]**2 instead of [-1,1]**2
    def _angles_interp(self, x, y, pow=1, eps=0):
        """Interpolates the angle factors separately so that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        angles = self.sidelengths
        ax = angles[0]
        bx = angles[2]
        ay = angles[3]
        by = angles[1]
        result1 = (ax*(1-y)**pow + bx*(1+y)**pow)/((1-y)**pow + (1+y)**pow)
        result2 = (ay*(1-x)**pow + by*(1+x)**pow)/((1-x)**pow + (1+x)**pow)
        return result1, result2

    def invtransform_v(self, v, pow=1):
        """
        Naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        anglex, angley = self._angles_interp(x, y, pow)
        sx = np.sin((1+x)*anglex/2)
        sy = np.sin((1+y)*angley/2)
        scx = np.sin((1-x)*anglex/2)
        scy = np.sin((1-y)*angley/2)
        a = scx * scy
        b = sx * scy
        c = sx * sy
        d = scx * sy
        mat = (np.stack([a, b, c, d], axis=-1) /
            (np.sin(anglex)* np.sin(angley))[..., np.newaxis] )
        result = (mat.dot(self.ctrlvector.T)).T
        result = self._fix_corners(x, y, result)
        return vectortoll(result)


class NSlerpQuad2(SphereProjection):
    #FIXME convert to [0,1]**2 instead of [-1,1]**2    
    def _angles_interp(self, x, y, pow=1, eps=0):
        """Interpolates the angle factor together that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        angles = self.sidelengths
        if np.max(angles) - np.min(angles) <= eps:
            return np.mean(angles)
        a = ((1-x)*(1-y)*(1+x))**pow
        b = ((1-y)*(1+x)*(1+y))**pow
        c = ((1-x)*(1+x)*(1+y))**pow
        d = ((1-x)*(1-y)*(1+y))**pow
        numer = a*angles[0] + b*angles[1] + c*angles[2] + d*angles[3]
        denom = a + b + c + d
        return numer/denom

    def invtransform_v(self, v, pow=1):
        """
        Variant naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        angle = self._angles_interp(x,y, pow)[..., np.newaxis]
        a = (1-x)*(1-y)
        b = (1+x)*(1-y)
        c = (1+x)*(1+y)
        d = (1-x)*(1+y)
        mat = (np.sin(np.stack([a, b, c, d], axis=-1)*angle/4) /
               np.sin(angle))
        result = (mat.dot(self.ctrlvector.T)).T
        result = self._fix_corners(x, y, result)
        return vectortoll(result)

class EllipticalQuad(SphereProjection):
    #FIXME convert to [0,1]**2 instead of [-1,1]**2    
    def __init__(self, ctrlpts, eps=1E-6):
        """Parameters:
        ctrlpts: 2x4 Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        sidelength = self.sidelengths
        assert abs(sidelength[0] - sidelength[2]) < eps
        assert abs(sidelength[1] - sidelength[3]) < eps
        vertangles = (np.roll(self.baz, -1) - self.faz) % 360
        assert abs((vertangles - vertangles.mean()).sum()) < eps

    def invtransform_v(self, v):
        """An extension of the elliptical map.
        """
        #FIXME needs rotations
        rot_base = self.ctrlvector
        a = rot_base[0,0]
        b = rot_base[0,1]
        c = rot_base[0,2]
        x = v[0]
        y = v[1]
        axt = (1 - a**2*x**2)
        byt = (1 - b**2*y**2)
        at = (1-a**2)
        bt = (1-b**2)
        u = a * x * sqrt(byt/bt)
        v = b * y * sqrt(axt/at)
        w = c * sqrt(axt*byt/(at*bt))
        result = np.stack([u,v,w], axis=0)
        result = self._fix_corners(x, y, result)
        return vectortoll(result)
#%%
class ConformalTri(MapProjection):
    pass

if __name__ == "__main__":
    import doctest
    sup = np.testing.suppress_warnings()
    sup.filter(RuntimeWarning)
    options = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    with sup:
        doctest.testmod(optionflags = options)
