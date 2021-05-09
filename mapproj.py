#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:38:02 2019

@author: brsr
"""

import geopandas
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon, Point
import pyproj
#import homography
import warnings
import numpy as np
from abc import ABC
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy.special import hyp2f1, gamma, ellipj, ellipk, ellipkinc


#TODO:
#vectorize all the things
#find a better implementation of conformal
#   (some kind of circle-packing thing?)
#repeated subdivision

#arange3 = np.arange(3)
#FIRST AXIS IS SPATIAL

TGTPTS3 = np.eye(3)
TGTPTS4 = np.array([[0, 1, 1, 0],
                    [0, 0, 1, 1]])

def normalize(vectors, axis=0):
    """Normalizes vectors in n-space. The zero vector remains the zero vector.
    Args:
        vectors: Array of vectors
        axis: Which axis to take the norm over (by default the first axis, 0)
    >>> x = np.stack((np.ones(5), np.arange(5)), axis=0)
    >>> normalize(x)
    array([[1.        , 0.70710678, 0.4472136 , 0.31622777, 0.24253563],
           [0.        , 0.70710678, 0.89442719, 0.9486833 , 0.9701425 ]])
    """
    n = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return np.where(n <= 0, 0, vectors / n)

def complex_to_float2d(arr):
    """Converts a complex array to a multidimensional float array.
    >>> x = np.exp(2j*np.pi*np.linspace(0, 1, 5)).round()
    >>> complex_to_float2d(x.round())
    array([[ 1.,  0.],
           [ 0.,  1.],
           [-1.,  0.],
           [-0., -1.],
           [ 1., -0.]])
    """
    return arr.view(float).reshape(list(arr.shape) + [-1])

def float2d_to_complex(arr):
    """Converts a multidimensional float array to a complex array.
    Input must be a float type, since there is no integer complex type.
    >>> y = np.arange(8, dtype=float).reshape((-1, 2))
    >>> float2d_to_complex(y)
    array([[0.+1.j],
           [2.+3.j],
           [4.+5.j],
           [6.+7.j]])
    """
    return arr.view(complex)

def sqrt(x):
    """Real sqrt clipped to 0 for negative values.

    >>> x = np.array([-np.inf, -1, 0, 1, np.inf, np.nan])
    >>> sqrt(x)
    array([ 0.,  0.,  0.,  1., inf, nan])
    """
    return np.where(x < 0, 0, np.sqrt(x))

def geodesics(lon, lat, geod, n=100, includepts=False):
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
    ctrlboundary = geopandas.GeoSeries(result)
    if includepts:
        controlpts = arraytoptseries(np.array([lon, lat]))
        ctrlpoly = geopandas.GeoSeries(pd.concat([ctrlboundary, controlpts],
                                            ignore_index=True))
        return ctrlpoly
    else:
        return ctrlboundary

def transform_antipode(lon, lat):
    """Transform a point given by lon and lat to its antipode."""
    lon2 = lon - 180
    np.where(lon2 <= -180, lon2 + 360, lon2)
    return lon2, -lat

def ptseriestoarray(ser):
    """Convert a geopandas GeoSeries containing shapely Points
    (or LineStrings of all the same length) to an array of
    shape (2, n) or (3, n).
    """
    return np.stack([x.coords for x in ser], axis=-1).squeeze()

def arraytoptseries(arr, crs={'epsg': '4326'}):
    """Convert an array of shape (2, ...) or (3, ...) to a
    geopandas GeoSeries containing shapely Point objects.
    """
    if arr.shape[0] == 2:
        result = geopandas.GeoSeries([Point(x[0], x[1])
                                    for x in arr.reshape(2, -1).T])
    else:
        result = geopandas.GeoSeries([Point(x[0], x[1], x[2])
                                    for x in arr.reshape(3, -1).T])
    #result.crs = crs
    return result

def transeach(func, geoms):
    """Transform each element of geoms using the function func."""
    plist = []
    for geom in geoms:
        if isinstance(geom, Point):
            #special logic for points
            ll = geom.coords[0]
            plist.append(Point(func(*ll)))
        else:
            plist.append(shapely.ops.transform(func, geom))
    return geopandas.GeoSeries(plist)

def graticule(spacing1=15, spacing2=1,
              lonrange = [-180, 180], latrange = [-90, 90]):
    """
    Create a graticule (or another square grid)
    """
    a = int((lonrange[1] - lonrange[0])//spacing2)
    b = int((latrange[1] - latrange[0])//spacing1)
    c = int((lonrange[1] - lonrange[0])//spacing1)
    d = int((latrange[1] - latrange[0])//spacing2)
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

def anglesgivensides(sides, scale=180/np.pi):
    """Given side lengths of a triangle, determines the interior angle at each
    vertex, and the radius of the circumcircle.
    >>> sides=np.array( [3,4,5])
    >>> anglesgivensides(sides)
    """
    #might be more stable to use law of cotangents, but eh
    r = np.product(sides)/sqrt(
            2*np.sum(sides**2*np.roll(sides,1)**2)
            -np.sum(sides**4))
    s1 = sides
    s2 = np.roll(sides, -1)
    s3 = np.roll(sides, 1)
    cosangle = (s2**2 + s3**2 - s1**2)/ (2*s2*s3)
    angles = np.arccos(cosangle)
    return angles*scale, r

def trigivenlengths(sides):
    """Given side lengths, creates the vertices of a triangle with those
    side lengths, and having circumcenter at 0,0.
    >>> sides=np.array( [3,4,5])
    >>> np.round(trigivenlengths(sides), decimals=8)
    array([[-2.5, -0.7,  2.5],
           [ 0. , -2.4,  0. ]])
    """
    angles, r = anglesgivensides(sides, scale=1)
    return r*trigivenangles(np.roll(angles, -1), scale=1)
#%%

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
    >>> x = np.stack((c, s, z), axis=0)
    >>> y = np.stack((c, z, s), axis=0)
    >>> np.round(central_angle(x, y)/np.pi*180)
    array([  0.,  60.,  90.,  60.,   0.])
    """
    cos = np.sum(x*y, axis=0)
    sin = np.linalg.norm(np.cross(x, y, axis=0), axis=0)
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

def dslerp(pt1, pt2, intervals):
    """The derivative of slerp."""
    t = intervals
    angle = central_angle(pt1, pt2)[..., np.newaxis]
    return (-np.cos((1 - t)*angle)*pt1 + np.cos(t*angle)*pt2)/np.sin(angle)

def triangle_solid_angle(a, b, c, axis=0):
    """Solid angle of a triangle with respect to 0. If vectors have norm 1,
    this is the spherical area. Note there are two solid angles defined by
    three points, determined by orientation of a, b, c.
    Formula is from Van Oosterom, A; Strackee, J (1983).
    "The Solid Angle of a Plane Triangle". IEEE Trans. Biom. Eng.
    BME-30 (2): 125–126. doi:10.1109/TBME.1983.325207.
    Args:
        a, b, c: Coordinates of points on the sphere.
    Returns: Array of solid angles.
    >>> t = np.linspace(0, np.pi, 5)
    >>> a = np.stack([np.cos(t), np.sin(t), np.zeros(5)],axis=0)
    >>> b = np.array([0, 1, 1])/np.sqrt(2)
    >>> c = np.array([0, -1, 1])/np.sqrt(2)
    >>> np.round(triangle_solid_angle(a, b, c), 4)
    array([ 1.5708,  1.231 ,  0.    , -1.231 , -1.5708])
    """
    axes = (axis,axis)
    top = np.tensordot(a, np.cross(b, c, axis=axis), axes=axes)
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    nc = np.linalg.norm(c, axis=0)
    bottom = (na * nb * nc + np.tensordot(a, b, axes=axes) * nc
              + np.tensordot(b, c, axes=axes) * na
              + np.tensordot(c, a, axes=axes) * nb)
    return 2 * np.arctan2(top, bottom)

def shoelace(pts):
    """Find area of polygon in the plane defined by pts, where pts is an
    array with shape (2,n).
    >>> pts = np.arange(6).reshape(2,-1)%4
    >>> shoelace(pts)
    2.0
    """
    return abs(np.sum(np.cross(pts, np.roll(pts, -1, axis=1), axis=0)))/2

def antipode_v(ll):
    """Antipodes of points given by longitude and latitude."""
    antipode = ll.copy()
    antipode[0] -= 180
    index = antipode[0] < -180
    antipode[0, index] += 360
    antipode[1] *= -1
    return antipode

def omegascale(adegpts, degpts_t, geod, spacing=1):
    """Estimate scale factor and max deformation angle for a map projection
    based on a grid of points
    """
    #actrlpts, tgtpts,
    #ar, p = geod.polygon_area_perimeter(actrlpts[0], actrlpts[1])
    #at = shoelace(tgtpts)
    es = geod.es
    a = geod.a
    factor = np.pi/180
    #lon = adegpts[0]*factor
    lat = adegpts[1]*factor
    x = degpts_t[0]
    y = degpts_t[1]
    dx = np.gradient(x, factor*spacing)
    dy = np.gradient(y, factor*spacing)
    dxdlat, dxdlon = dx
    dydlat, dydlon = dy
    J = (dydlat*dxdlon - dxdlat*dydlon)
    R = a*np.sqrt(1-es)/(1-es*np.sin(lat)**2)
    h = sqrt((dxdlat)**2 + (dydlat)**2)*(1-es*np.sin(lat)**2)**(3/2)/(a*(1-es))
    k = sqrt((dxdlon)**2 + (dydlon)**2)*(1-es*np.sin(lat)**2)**(1/2)/(a*np.cos(lat))
    scale = J/(R**2*np.cos(lat))
    sinthetaprime = np.clip(scale/(h*k), -1, 1)
    aprime = sqrt(h**2 + k**2 + 2*h*k*sinthetaprime)
    bprime = sqrt(h**2 + k**2 - 2*h*k*sinthetaprime)
    sinomegav2 = np.clip(bprime/aprime, -1, 1)
    omega = 360*np.arcsin(sinomegav2)/np.pi
    return omega, scale

def rodrigues(center, v, theta):
    """Rodrigues formula: rotate vector v around center by angle theta
    """
    cxv = np.cross(center, v)
    cv = np.sum(center* v, axis=-1, keepdims=True)
    cc = v*np.cos(theta) + cxv*np.sin(theta) + center*cv*(1-np.cos(theta))
    return cc


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
        shape = [-1, ] + list(pts.shape[1:])
        return result.T.reshape(shape)

    def invtransform_v(self, pts, **kwargs):
        rpts = pts.reshape((pts.shape[0],-1)).T
        result = []
        for xy in rpts:
            result.append(self.invtransform(*xy, **kwargs))
        result = np.array(result)
        shape = [-1, ] + list(pts.shape[1:])
        return result.T.reshape(shape)
#%%
class UV(Projection):
    nctrlpts = 4
    @staticmethod
    def grid(**kwargs):
        """Create a square grid"""
        return graticule(spacing1=1, spacing2=0.01,
                         lonrange=[0,1], latrange=[0,1])

    @staticmethod
    def gridpolys(n=11):
        poi = np.array(np.meshgrid(np.linspace(0, 1, n),
                                   np.linspace(0, 1, n)))
        poilist = []
        for i, j in np.ndindex(n-1,n-1):
            x = Polygon([poi[:, i, j],      poi[:, i, j+1],
                         poi[:, i+1, j+1],  poi[:, i+1, j]])
            poilist.append(x)
        poiframe = geopandas.geoseries.GeoSeries(poilist)
        return poiframe

    @staticmethod
    def segment(uv):
        u, v = uv
        index1 = u > v
        index2 = u < 1 - v
        #1 and 2 = 0
        #1 and not 2 = 1
        #not 1 and not 2 = 2
        #not 1 and 2 = 3
        result = np.zeros(u.shape)
        result[index1 & ~index2] = 1
        result[~index1 & ~index2] = 2
        result[~index1 & index2] = 3
        return result

class Bilinear(UV):
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

class Homeomorphism(UV):
    """Homeomorphism"""
    def __init__(self, tgtpts):
        self.tgtpts = tgtpts

class Barycentric(Projection):
    """Transforms between plane and barycentric coordinates"""
    nctrlpts = 3

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

    @staticmethod
    def grid(spacing1=0.1, spacing2=1E-2, rang = [0, 1], eps=1E-8):
        """Create a triangle grid in barycentric coordinates
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
        return grat

    @staticmethod
    def gridpolys(n=11, eps=0.01):
        poi = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        poi.append(1 - poi[0] - poi[1])
        poi = np.array(poi)
        poilist = []
        for i,j in np.ndindex(n-1,n-1):
            if poi[2, i, j] >= eps:
                x = Polygon([poi[:, i, j],poi[:, i, j+1],poi[:, i+1, j]])
                poilist.append(x)
            if poi[2, i+1, j+1] >= -eps:
                y = Polygon([poi[:, i+1, j+1],poi[:, i+1, j],poi[:, i, j+1]])
                poilist.append(y)
        poiframe = geopandas.geoseries.GeoSeries(poilist)
        return poiframe

    @staticmethod
    def segment(bary):
        return np.argmin(bary, axis=0)

class UnitVector(Projection):
    """Convert longitude and latitude to unit vector normals.
    The methods of this class are static, and mostly organized in a class
    for consistency."""

    @staticmethod
    def transform(x, y, **kwargs):
        pts = np.stack([x,y])
        vresult = UnitVector.transform_v(pts, **kwargs)
        return vresult

    @staticmethod
    def invtransform(x, y, z, **kwargs):
        pts = np.stack([x,y,z])
        vresult = UnitVector.invtransform_v(pts, **kwargs)
        return vresult

    @staticmethod
    def transform_v(ll, scale=np.pi/180):
        """Convert longitude and latitude to 3-vector

        >>> ll = np.arange(6).reshape(2,3)*18
        >>> UnitVector.transform_v(ll)
        array([[5.87785252e-01, 2.93892626e-01, 4.95380036e-17],
               [0.00000000e+00, 9.54915028e-02, 3.59914664e-17],
               [8.09016994e-01, 9.51056516e-01, 1.00000000e+00]])
        """
        lon, lat = ll*scale
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
        return np.stack([x, y, z], axis=0)

    @staticmethod
    def invtransform_v(pts, scale=180/np.pi):
        """Convert 3-vector to longitude and latitude.
        Vector does not have to be normalized.

        >>> UnitVector.invtransform_v(np.eye(3))
        array([[ 0., 90.,  0.],
               [ 0.,  0., 90.]])
        """
        lat = scale*np.arctan2(pts[2], sqrt(pts[1]**2 + pts[0]**2))
        lon = scale*np.arctan2(pts[1], pts[0])
        return np.stack([lon, lat], axis=0)

_unitsphgeod = pyproj.Geod(a=1, b=1)
class CtrlPtsProjection(Projection, ABC):
    """Subclass for any map projection that uses (2 or more) control points."""
    def __init__(self, ctrlpts, geod = _unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 or 2x4 Numpy array, latitude and longitude of
            each control point
        geod= a pyproj.Geod object. For a unit sphere use
            pyproj.Geod(a=1,b=1)
        """
        n = ctrlpts.shape[1]
        if self.nctrlpts != n:
            raise ValueError(
                'ctrlpts has wrong number of points for this projection')
        self.geod = geod
        #it's possible to get a geod where this would give the wrong answer,
        #but I think it would have to be really weird
        area, _ = geod.polygon_area_perimeter([0,120,-120],[0,0,0])
        self.totalarea = 2*area

        self.ctrlpts = ctrlpts
        ctrlpts_v = UnitVector.transform_v(ctrlpts)
        self.ctrlpts_v = ctrlpts_v
        center_v = ctrlpts_v.sum(axis=1)
        self.center_v = center_v / np.linalg.norm(center_v)
        self.center = UnitVector.invtransform_v(center_v)
        antipode = antipode_v(ctrlpts)
        self.antipode = antipode
        self.antipode_v = UnitVector.transform_v(antipode)
        self.sa = 0
        if self.nctrlpts > 2:
            faz, baz, sides = self.geod.inv(ctrlpts[0], ctrlpts[1],
                                              np.roll(ctrlpts[0], -1),
                                              np.roll(ctrlpts[1], -1))
            self.sides = sides
            self.faz = faz
            self.baz = baz
            self.ctrl_angles = (faz - np.roll(baz, 1))%360
            area, _ = geod.polygon_area_perimeter(*ctrlpts)
            self.area = area
            self.ca = central_angle(ctrlpts_v,
                                        np.roll(ctrlpts_v, -1, axis=1))
            for i in range(1, self.nctrlpts-1):
                self.sa += triangle_solid_angle(ctrlpts_v[..., 0],
                                                    ctrlpts_v[..., i],
                                                    ctrlpts_v[..., i+1])

            self.edgenormals = np.cross(ctrlpts_v,
                                        np.roll(ctrlpts_v, -1, axis=1), axis=0)

        else:
            faz, baz, sides = self.geod.inv(ctrlpts[0,0], ctrlpts[1,0],
                                              ctrlpts[0,1], ctrlpts[1,1])
            self.sides = sides
            self.faz = faz
            self.baz = baz
            self.area = 0
            self.ca = central_angle(ctrlpts_v[..., 0], ctrlpts_v[..., 1])
            self.edgenormals = np.cross(ctrlpts_v[..., 0], ctrlpts_v[..., 1])

        self.cosca = np.cos(self.ca)
        self.sinca = np.sin(self.ca)

        if self.sa < 0:
            warnings.warn('control polygon is in negative orientation, '
                          + 'may cause unusual results')

        if self.nctrlpts == 4:
            ctrlpts_v = self.ctrlpts_v
            v0 = ctrlpts_v[..., 0]
            v1 = ctrlpts_v[..., 1]
            v2 = ctrlpts_v[..., 2]
            v3 = ctrlpts_v[..., 3]
            poip1 = np.cross(np.cross(v0, v1), np.cross(v3, v2))
            poip2 = np.cross(np.cross(v0, v3), np.cross(v1, v2))
            poip = np.stack([[poip1, -poip1],
                             [poip2, -poip2]]).transpose(2,0,1)
            poip = poip / np.linalg.norm(poip, axis=0)
            self.poi_v = poip
            self.poi = UnitVector.invtransform_v(poip)
            self.crossx = np.cross(ctrlpts_v,
                                   np.roll(ctrlpts_v, -2, axis=1),
                                   axis=0)[..., :2]

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

    def lune(self, lon, lat):
        """
        Determine which lune a point or series of points lies in.
        Lune 0 is the lune with vertex at the centroid and edges passing through
        control points 0 and 1. Lune 1 is the same using control pts 1 and 2,
        and Lune 2 uses control pts 2 and 0.
        """
        #inexact on ellipsoids but close enough
        testpt = UnitVector.transform(lon, lat)
        testpt_v = testpt.reshape(3,-1)
        ctrlpts_v = self.ctrlpts_v
        center_v = self.center_v
        cx = np.cross(center_v, ctrlpts_v, axis=0)
        sk = cx.T @ testpt_v
        sg = sk >= 0
        ind = sg & ~np.roll(sg, shift=-1, axis=0)
        result = np.argmax(ind, axis=0)
        return result.reshape(testpt.shape[1:])

class BarycentricMapProjection(CtrlPtsProjection):
    nctrlpts = 3
    tweak = False
    bcenter = np.ones(3)/3

    def fixbary(self, bary):
        if self.tweak:
            return self.fixbary_normalize(bary)
        else:
            return self.fixbary_subtract(bary)

    @staticmethod
    def fixbary_normalize(bary):
        """Converts array bary to an array with sum = 1 by dividing by
        bary.sum(). Will return nan if bary.sum() == 0.

        >>> fixbary_normalize(np.arange(3))
        array([0.        , 0.33333333, 0.66666667])
        """
        bary = np.array(bary)
        return bary / np.sum(bary, axis=0, keepdims=True)

    @staticmethod
    def fixbary_subtract(bary):
        """Converts array bary to an array with sum = 1 by subtracting
        (bary.sum() - 1)/bary.shape[0].

        >>> fixbary_subtract(np.arange(3))
        array([-0.66666667,  0.33333333,  1.33333333])
        """
        bary = np.array(bary)
        s = (np.sum(bary, axis=0, keepdims=True) - 1)/bary.shape[0]
        return bary - s

    def _fix_corners(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        #print(lon, lat, ctrlpts, result)
        #print(index0.shape, result.shape, np.array([1, 0, 0])[..., np.newaxis].shape)
        result[..., index0] = np.array([1, 0, 0])[..., np.newaxis]
        result[..., index1] = np.array([0, 1, 0])[..., np.newaxis]
        result[..., index2] = np.array([0, 0, 1])[..., np.newaxis]
        return result

    def _fix_corners_inv(self, bary, result):
        index0 = (bary[0] == 1)
        index1 = (bary[1] == 1)
        index2 = (bary[2] == 1)
        if np.any(index0):
            result[..., index0] = self.ctrlpts_v[..., 0, np.newaxis]
        if np.any(index1):
            result[..., index1] = self.ctrlpts_v[..., 1, np.newaxis]
        if np.any(index2):
            result[..., index2] = self.ctrlpts_v[..., 2, np.newaxis]
        return result

class UVMapProjection(CtrlPtsProjection):
    nctrlpts = 4
    bcenter = np.ones(2)/2

    def _fix_corners(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        index3 = (lon == ctrlpts[0,3]) & (lat == ctrlpts[1,3])
        result[..., index0] = np.array([ 0,  0])[..., np.newaxis]
        result[..., index1] = np.array([ 1,  0])[..., np.newaxis]
        result[..., index2] = np.array([ 1,  1])[..., np.newaxis]
        result[..., index3] = np.array([ 0,  1])[..., np.newaxis]
        return result

    def _fix_corners_inv(self, x, y, result):
        index0 = (x == 0) & (y == 0)
        index1 = (x == 1) & (y == 0)
        index2 = (x == 1) & (y == 1)
        index3 = (x == 0) & (y == 1)
        if np.any(index0):
            result[..., index0] = self.ctrlpts_v[..., 0, np.newaxis]
        if np.any(index1):
            result[..., index1] = self.ctrlpts_v[..., 1, np.newaxis]
        if np.any(index2):
            result[..., index2] = self.ctrlpts_v[..., 2, np.newaxis]
        if np.any(index3):
            result[..., index3] = self.ctrlpts_v[..., 3, np.newaxis]
        return result

#%% not-polygonal projections
class ChambTrimetric(CtrlPtsProjection):
    """Chamberlin trimetric projection"""
    #FIXME this implementation fails for control triangles with 
    #high aspect ratios
    nctrlpts = 3

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod)
        self.tgtpts = trigivenlengths(self.sides)
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

class LinearTrimetric(CtrlPtsProjection):
    """The linear variation of the Chamberlin Trimetric projection."""
    nctrlpts = 3
    matrix1 = np.array([[0,-1],
               [1,0]])
    matrix2 = np.array([[0, -1, 1],
               [1, 0, -1],
               [-1, 1, 0]])
    matrixinv1 = np.array([[-2,1,1],
              [1,-2,1],
              [1,1,-2]])*2/3

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        self.radius = ((geod.a**(3/2) + geod.b**(3/2))/2)**(2/3)
        self.tgtpts = trigivenlengths(self.sides)
        self.setmat()
        # try:
        #     self.orienttgtpts(self.tgtpts)
        #     self.setmat()
        # except ValueError:
        #     pass

        vctrl = self.ctrlpts_v
        self.invctrlvector = np.linalg.pinv(vctrl)
        self.invperpmatrix = self.invctrlvector @ self.invctrlvector.T
        cosrthmin = 1 / np.sqrt(self.invperpmatrix.sum())
        self.hminall = np.arccos(cosrthmin)**2

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
        print('k: ', k)
        #hmax = np.pi**2-np.max(k, axis=0)
        hminall = self.hminall
        h = np.where(hmin < hminall, hminall, hmin)
        print('h: ', h)
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
            print('delta:', delta)
            print('h: ', h)
            if np.max(np.abs(delta)) < stop:
                break
        #h = np.clip(h, hmin, hmax)
        rsq = np.clip(k + h, 0, np.pi**2)
        c = np.cos(np.sqrt(rsq))
        vector = self.invctrlvector.T @ c
        print(c)
        print(vector)
        return UnitVector.invtransform_v(vector).reshape(pts.shape)

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

class Alfredo(BarycentricMapProjection):
    """this doesn't really accomplish anything"""

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        self.cosADfactor = (np.cross(np.roll(ctrlpts_v, 1, axis=1),
                            np.roll(ctrlpts_v, -1, axis=1), axis=0) +
                            ctrlpts_v * np.linalg.det(ctrlpts_v))
        self.tweak = tweak

    def transform_v(self, ll):
        rll = ll.reshape(2, -1)
        ctrlpts_v = self.ctrlpts_v
        cosADfactor = self.cosADfactor
        vtestpt = UnitVector.transform_v(rll)
        cosAPi = (vtestpt.T @ ctrlpts_v).T
        cosADi = (vtestpt.T @ cosADfactor).T
        pli = np.sqrt((1-cosAPi)/(1-cosADi))
        b = 1 - pli
        result = self.fixbary(b)
        shape = (3,) + ll.shape[1:]
        return result.reshape(shape)

    def invtransform(self, *args, **kwargs):
        return NotImplemented

#%%
class Areal(BarycentricMapProjection):
    """Spherical areal projection."""

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod: a pyproj.Geod object. For a unit sphere use
                pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        a_i = np.sum(np.roll(self.ctrlpts_v, -1, axis=1) *
                          np.roll(self.ctrlpts_v, 1, axis=1), axis=0)
        self.a_i = a_i
        self.b_i = (np.roll(a_i, -1) + np.roll(a_i, 1))/(1+a_i)
        self.tau_c = self.tau(self.area)

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
            return areas/self.area
        except ValueError:
            raise TypeError()

    def invtransform_v(self, bary):
        rbary = bary.reshape(3,-1)
        if not self.geod.sphere:
            warnings.warn('inverse transform is approximate on ellipsoids')
        b_i = self.b_i[:,np.newaxis]
        tau = self.tau_c
        tau_i = self.tau(self.area*rbary)
        t_i = tau_i/tau
        c_i = t_i / ((1+b_i) + (1-b_i) * t_i)
        f_i = c_i / (1 - np.sum(c_i, axis=0))
        vector = self.ctrlpts_v @ f_i
        shape = [2] + list(bary.shape[1:])
        result = UnitVector.invtransform_v(vector).reshape(shape)
        return result
#%%
class BisectTri(BarycentricMapProjection):
    """Inverse is only approximate
    """
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        #v_0 = ctrlpts_v[..., 0]
        #v_1 = ctrlpts_v[..., 1]
        #v_2 = ctrlpts_v[..., 2]
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        aream = []
        for i in range(3):
            #index = np.roll(np.arange(3), -i)[:2]
            #lona = list(ctrlpts[0, index]) + [self.midpoint[0,i],]
            #lata = list(ctrlpts[1, index]) + [self.midpoint[1,i],]
            #am, _ = self.geod.polygon_area_perimeter(lona, lata)
            am = triangle_solid_angle(ctrlpts_v[:,i], ctrlpts_v[:,(i+1)%3],
                                      midpoint_v[:,i])
            #vc[:,0], mi, lproj)
            aream.append(am)
        self.aream = np.array(aream)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        areas = []
        vctrlpts = self.ctrlpts_v
        actrlpts = self.ctrlpts
        geod = self.geod
        area = self.area
        for i in range(3):
            vc = np.roll(vctrlpts, i, axis=1)
            #ac = np.roll(actrlpts, i, axis=1)
            mi = self.midpoint_v[:,-i]#?
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            #lllproj = UnitVector.invtransform_v(lproj)
            #loni = [ac[0,0], mi[0], lllproj[0]]
            #lati = [ac[1,0], mi[1], lllproj[1]]
            #a1, _ = geod.polygon_area_perimeter(loni, lati)
            a1 = triangle_solid_angle(vc[:,0], mi, lproj)
            areas.append(a1)
        areas = np.array(areas) + self.aream
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
        b1 + 0
        beta = np.array([b1,b2,b3])
        vctrlpts3 = self.ctrlpts_v

        #xs = []
        ptts = []
        for i in range(3):
            beta1, beta2, beta3 = np.roll(beta, -i, axis=0)
            x = beta2/(1 - beta1)
            #xs.append(x)
            a = x * self.area
            pt0 = vctrlpts3[:,i]
            pt1 = vctrlpts3[:,i-2]
            pt2 = vctrlpts3[:,i-1]
            cosw = pt1 @ pt2
            w = np.arccos(cosw)
            sinw = np.sin(w)
            p2 = ((np.cos(a/2)* pt2 @ np.cross(pt0, pt1)- np.sin(a/2)*pt2 @ (pt1 + pt0))
                  + np.sin(a/2)*cosw*(1 + pt1 @ pt0))
            p3 = sinw*np.sin(a/2)*(1 + pt0 @ pt1)
            r = 2*p3*p2/(p2**2 - p3**2)
            t = np.arctan(r)/w#really close to just x
            #print(x, t)
            #t = x
            ptt = slerp(pt2, pt1, t)
            ptts.append(ptt)

        ptts = np.array(ptts).T
        ns = np.cross(vctrlpts3, ptts, axis=0)
        pts = np.cross(ns, np.roll(ns, -1, axis=1), axis=0)
        v = pts.sum(axis=1)
        v = self._fix_corners_inv(beta, v)
        return UnitVector.invtransform_v(v)

class BisectTri2(BarycentricMapProjection):
    """Inverse is only approximate"""

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        #v_0 = ctrlpts_v[..., 0]
        #v_1 = ctrlpts_v[..., 1]
        #v_2 = ctrlpts_v[..., 2]
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        aa = []
        vctrlpts = self.ctrlpts_v
        actrlpts = self.ctrlpts
        for i in range(3):
            vc = np.roll(vctrlpts, i, axis=1)
            ac = np.roll(actrlpts, i, axis=1)
            mi = self.midpoint[:,-i]
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            lllproj = UnitVector.invtransform_v(lproj)
            dist1x = central_angle(vc[..., 1], lproj)
            f, b, dist1x = self.geod.inv(mi[0], mi[1],
                                         lllproj[0],lllproj[1])
            f0, b0, _ = self.geod.inv(mi[0], mi[1],
                                      ac[0,2], ac[1,2])
            deltaf = (f-f0) % 360
            if (deltaf <= 90) | (deltaf > 270):
                s = 1
            else:
                s = -1
            t = s*dist1x/self.sides[i] + 1/2
            #print(t)
            aa.append(t)
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
        b1 + 0
        beta = np.array([b1,b2,b3])
        vctrlpts3 = self.ctrlpts_v

        #xs = []
        ptts = []
        for i in range(3):
            beta1, beta2, beta3 = np.roll(beta, -i, axis=0)
            x = beta2/(1 - beta1)
            pt1 = vctrlpts3[:,i-2]
            pt2 = vctrlpts3[:,i-1]
            ptt = slerp(pt2, pt1, x)
            ptts.append(ptt)

        ptts = np.array(ptts).T
        ns = np.cross(vctrlpts3, ptts, axis=0)
        pts = np.cross(ns, np.roll(ns, -1, axis=1), axis=0)
        v = pts.sum(axis=1)
        v = self._fix_corners_inv(beta, v)
        return UnitVector.invtransform_v(v)

class FullerEq(BarycentricMapProjection):

    def transform_v(self, ll):
        vtestpt_pre = UnitVector.transform(*ll)
        vtestpt = vtestpt_pre.reshape(3,-1)
        ctrlpts_v = self.ctrlpts_v
        b = []
        for i in range(3):
            v0 = ctrlpts_v[..., i]
            v1 = ctrlpts_v[..., (i+1)%3]
            v2 = ctrlpts_v[..., (i-1)%3]
            cosw01 = v0 @ v1
            cosw02 = v0 @ v2
            w01 = np.arccos(cosw01)
            w02 = np.arccos(cosw02)
            w = (w01 + w02) / 2
            sinw = np.sin(w)
            cosw = np.cos(w)
            vt01 = np.tensordot(vtestpt, np.cross(v0, v1), axes=(0,0))
            vt12 = np.tensordot(vtestpt, np.cross(v1, v2), axes=(0,0))
            vt20 = np.tensordot(vtestpt, np.cross(v2, v0), axes=(0,0))
            bi = np.arctan2(sinw*vt12, cosw*vt12 + vt01 + vt20)/w
            #gx = vt12 + cosw*(vt01 + vt20)
            #tx = np.arctan2(sinw*(vt20 + vt01),gx)/w
            b.append(bi)
            #b.append(1-tx)
        b = np.array(b)
        result = self.fixbary_subtract(b)
        return result.reshape(vtestpt_pre.shape)

    def invtransform(self, b1, b2, b3):
        b1 + 0 #still not vectorized
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]

        w = self.ca.mean()
        bi = np.array([b1, b2, b3])
        cw = np.cos(w)
        #sw = np.sin(w)
        cbw = np.cos(bi*w)
        sbw = np.sin(bi*w)
        pcbw = np.product(cbw)
        psbw = np.product(sbw)
        scc = np.sum(sbw * np.roll(cbw, -1) * np.roll(cbw, 1))
        css = np.sum(cbw*np.roll(sbw, -1)*np.roll(sbw, 1))
        objw2 = np.array([2*pcbw - cw - 1,
                          2*scc,
                          3*pcbw + 3 - css,
                          2*psbw])
        rts = np.roots(objw2)[-1]#FIXME solve this cubic explicitly
        rts = rts.real
        k = np.arctan(rts)/w
        #f0 = np.where(bi[0] + k > 1, -1, 1)
        f1 = np.where(bi[1] + k > 1, -1, 1)
        f2 = np.where(bi[2] + k > 1, -1, 1)
        #v01 = slerp(v1, v0, bi[0] + k)
        #v02 = slerp(v2, v0, bi[0] + k)
        #cx12 = np.cross(v01, v02)*f0
        v12 = slerp(v2, v1, bi[1] + k)
        v10 = slerp(v0, v1, bi[1] + k)
        cx20 = np.cross(v12, v10)*f1
        v20 = slerp(v0, v2, bi[2] + k)
        v21 = slerp(v1, v2, bi[2] + k)
        cx01 = np.cross(v20, v21)*f2

        v0x = normalize(np.cross(cx20, cx01))
        #v1x = normalize(np.cross(cx01, cx12))
        #v2x = normalize(np.cross(cx12, cx20))
        v0x = self._fix_corners_inv(bi, v0x)
        #print(v0x)
        return UnitVector.invtransform_v(v0x)

class Fuller(BarycentricMapProjection):

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.tweak = tweak

    def transform(self, lon, lat):
        lon + 0#will TypeError if lon is not a number
        vtestpt = UnitVector.transform(lon, lat)
        ctrlpts_v = self.ctrlpts_v
        b = []
        for i in range(3):
            v0 = ctrlpts_v[..., i]
            v1 = ctrlpts_v[..., (i+1)%3]
            v2 = ctrlpts_v[..., (i+2)%3]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt20 = vtestpt @ np.cross(v2, v0)
            cosw01 = v0 @ v1
            cosw02 = v0 @ v2
            w01 = np.arccos(cosw01)
            w02 = np.arccos(cosw02)
            if np.isclose(w01, w02):
                w = (w01 + w02) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                g = vt12 + cosw*(vt01 + vt20)
                ti = self._b_eq(w, sinw, vt20, vt01, g)
            else:
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
        if self.tweak:
            return self._invtransform_normalize(b1, b2, b3)
        else:
            return self._invtransform_subtract(b1, b2, b3)

    def _invtransform_subtract(self, b1, b2, b3):
        b1 + 0#will TypeError if not a number
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        def objective(k):
            f0 = np.where(bi[0] + k > 1, -1, 1)
            f1 = np.where(bi[1] + k > 1, -1, 1)
            f2 = np.where(bi[2] + k > 1, -1, 1)
            v01 = slerp(v1, v0, bi[0] + k)
            v02 = slerp(v2, v0, bi[0] + k)
            cx12 = np.cross(v01, v02)*f0
            v12 = slerp(v2, v1, bi[1] + k)
            v10 = slerp(v0, v1, bi[1] + k)
            cx20 = np.cross(v12, v10)*f1
            v20 = slerp(v0, v2, bi[2] + k)
            v21 = slerp(v1, v2, bi[2] + k)
            cx01 = np.cross(v20, v21)*f2

            v0x = normalize(np.cross(cx20, cx01))
            v1x = normalize(np.cross(cx01, cx12))
            v2x = normalize(np.cross(cx12, cx20))
            #this is slightly more robust than the triple product
            return (np.linalg.norm(v0x-v1x)
                    + np.linalg.norm(v1x-v2x)
                    + np.linalg.norm(v2x-v0x))
            # dv01 = dslerp(v1, v0, bi[0] + k)
            # dv02 = dslerp(v2, v0, bi[0] + k)
            # dcx12 = (np.cross(dv01, v02) + np.cross(v01, dv02))*f0
            # dv12 = dslerp(v2, v1, bi[1] + k)
            # dv10 = dslerp(v0, v1, bi[1] + k)
            # dcx20 = (np.cross(dv12, v10) + np.cross(v12, dv10))*f1
            # dv20 = dslerp(v0, v2, bi[2] + k)
            # dv21 = dslerp(v1, v2, bi[2] + k)
            # dcx01 = (np.cross(dv20, v21) + np.cross(v20, dv21))*f2

            # derivative = dcx12 @ v0x + dcx20 @ v1x + dcx01 @ v2x
            # return cx12 @ v0x, derivative
        if b1 == 0 or b2 == 0 or b3 == 0:
            k = 0
        elif np.allclose(self.sides, np.roll(self.sides, 1)):
            kx = self._k_eq(b1, b2, b3)
            k = kx[2]#FIXME is 2 always the right one?
        else:
            #FIXME why is this so freakin slow
            res = minimize_scalar(objective, bracket=[0,0.1])
            k = res.x
        #f0 = np.where(bi[0] + k > 1, -1, 1)
        f1 = np.where(bi[1] + k > 1, -1, 1)
        f2 = np.where(bi[2] + k > 1, -1, 1)
        #v01 = slerp(v1, v0, bi[0] + k)
        #v02 = slerp(v2, v0, bi[0] + k)
        #cx12 = np.cross(v01, v02)*f0
        v12 = slerp(v2, v1, bi[1] + k)
        v10 = slerp(v0, v1, bi[1] + k)
        cx20 = np.cross(v12, v10)*f1
        v20 = slerp(v0, v2, bi[2] + k)
        v21 = slerp(v1, v2, bi[2] + k)
        cx01 = np.cross(v20, v21)*f2

        v0x = normalize(np.cross(cx20, cx01))
        #v1x = normalize(np.cross(cx01, cx12))
        #v2x = normalize(np.cross(cx12, cx20))
        v0x = self._fix_corners_inv(bi, v0x)
        return UnitVector.invtransform_v(v0x)

    def _k_eq(self, b1, b2, b3):
        w = self.ca.mean()
        bi = np.array([b1, b2, b3])
        cw = np.cos(w)
        #sw = np.sin(w)
        cbw = np.cos(bi*w)
        sbw = np.sin(bi*w)
        pcbw = np.product(cbw)
        psbw = np.product(sbw)
        scc = np.sum(sbw * np.roll(cbw, -1) * np.roll(cbw, 1))
        css = np.sum(cbw*np.roll(sbw, -1)*np.roll(sbw, 1))
        objw2 = np.array([2*pcbw - cw - 1,
                          2*scc,
                          3*pcbw + 3 - css,
                          2*psbw])
        rts = np.roots(objw2)
        return np.arctan(rts)/w

    def _invtransform_normalize(self, b1, b2, b3):
        b1 + 0#will TypeError if not a number
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        def objective(k):
            f0 = np.where(bi[0] * k > 1, -1, 1)
            f1 = np.where(bi[1] * k > 1, -1, 1)
            f2 = np.where(bi[2] * k > 1, -1, 1)
            v01 = slerp(v1, v0, bi[0] * k)
            v02 = slerp(v2, v0, bi[0] * k)
            cx12 = normalize(np.cross(v01, v02))*f0
            v12 = slerp(v2, v1, bi[1] * k)
            v10 = slerp(v0, v1, bi[1] * k)
            cx20 = normalize(np.cross(v12, v10))*f1
            v20 = slerp(v0, v2, bi[2] * k)
            v21 = slerp(v1, v2, bi[2] * k)
            cx01 = normalize(np.cross(v20, v21))*f2
            v0x = normalize(np.cross(cx20, cx01))
            v1x = normalize(np.cross(cx01, cx12))
            v2x = normalize(np.cross(cx12, cx20))
            #i think this is slightly more robust than the triple product
            return (np.linalg.norm(v0x-v1x)
                    + np.linalg.norm(v1x-v2x)
                    + np.linalg.norm(v2x-v0x))

        res = minimize_scalar(objective, bracket=[1,1.1])
        k = res.x
        #f0 = np.where(bi[0] * k > 1, -1, 1)
        f1 = np.where(bi[1] * k > 1, -1, 1)
        f2 = np.where(bi[2] * k > 1, -1, 1)

        v12 = slerp(v2, v1, bi[1] * k)
        v10 = slerp(v0, v1, bi[1] * k)
        cx20 = normalize(np.cross(v12, v10))*f1
        v20 = slerp(v0, v2, bi[2] * k)
        v21 = slerp(v1, v2, bi[2] * k)
        cx01 = normalize(np.cross(v20, v21))*f2

        v0x = normalize(np.cross(cx20, cx01))
        v0x = self._fix_corners_inv(bi, v0x)
        return UnitVector.invtransform_v(v0x)

class SnyderEA(BarycentricMapProjection):

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        v_0 = ctrlpts_v[..., 0]
        v_1 = ctrlpts_v[..., 1]
        v_2 = ctrlpts_v[..., 2]
        self.v_01 = v_0 @ v_1
        self.v_12 = v_1 @ v_2
        self.v_20 = v_2 @ v_0
        self.v_012 = np.linalg.det(ctrlpts_v)
        self.c = self.v_12
        self.c2 = self.c**2
        self.s2 = 1 - self.c2
        self.s = sqrt(self.s2)
        self.w = np.arccos(self.c)
        self.midpoint_v = v_1 + v_2
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        lona = list(ctrlpts[0,:2]) + [self.midpoint[0],]
        lata = list(ctrlpts[1,:2]) + [self.midpoint[1],]
        self.area01m, _ = self.geod.polygon_area_perimeter(lona, lata)

    def transform(self, lon, lat):
        lon + 0
        actrlpts = self.ctrlpts
        ctrlpts_v = self.ctrlpts_v
        area = self.area
        geod = self.geod
        vtestpt = UnitVector.transform(lon, lat)
        lproj = -np.cross(np.cross(ctrlpts_v[..., 1], ctrlpts_v[..., 2]),
                          np.cross(ctrlpts_v[..., 0], vtestpt))
        norm = np.linalg.norm(lproj, axis=0, keepdims=True)
        if norm != 0:
            lproj = lproj / norm
        lllproj = UnitVector.invtransform_v(lproj)
        cosAP = ctrlpts_v[..., 0] @ vtestpt
        cosAD = ctrlpts_v[..., 0] @ lproj
        pl = sqrt((1-cosAP)/(1-cosAD))
        b0 = 1 - pl
        lona = [actrlpts[0,0], self.midpoint[0], lllproj[0]]
        lata = [actrlpts[1,0], self.midpoint[1], lllproj[1]]
        a1, _ = geod.polygon_area_perimeter(lona, lata)
        a1 += self.area01m
        b2 = a1/area * pl
        b1 = 1 - b0 - b2
        result = np.stack([b0,b1,b2])
        bresult = self._fix_corners(lon, lat, result)
        return np.where(np.isfinite(bresult), bresult, 0)

    def invtransform(self, b1, b2, b3):
        ctrlpts_v = self.ctrlpts_v
        area = self.area
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
        result = 1/w*np.arctan2(Gp, Fp)
        vd = slerp(ctrlpts_v[..., 1], ctrlpts_v[..., 2], result)
        AdotD = ctrlpts_v[..., 0] @ vd
        AdotP = 1 - lp**2*(1-AdotD)
        t = np.arccos(AdotP)/np.arccos(AdotD)
        vresult = slerp(ctrlpts_v[..., 0], vd, t)
        bary = np.stack([b1, b2, b3])
        vresult = self._fix_corners_inv(bary, vresult)
        vresult[~np.isfinite(vresult)] = 0
        return UnitVector.invtransform_v(vresult)

class SnyderEA3(BarycentricMapProjection):
    tmat = np.array([[1/3,0,0],
                     [1/3,1,0],
                     [1/3,0,1]])
    tmatinv = np.array([[3,0,0],
                        [-1,1,0],
                        [-1,0,1]])

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        subproj = []
        #want the center that divides the triangle into 3 equal-area triangles
        ap = Areal(ctrlpts)
        center = ap.invtransform(1/3, 1/3, 1/3)
        self.center = center
        self.center_v = UnitVector.transform(*center)
        arr = np.arange(3)
        for i in range(3):
            index = np.roll(arr, -i)[1:]
            cp = np.concatenate([center[:,np.newaxis],
                                 ctrlpts[:, index]], axis=1)
            pj = SnyderEA(cp)
            subproj.append(pj)
        self.subproj = subproj

    def transform(self, lon, lat):
        subproj = self.subproj
        i = self.lune(lon, lat)
        pj = subproj[i-1]#shift because we want the opposite vertex
        betap = pj.transform(lon, lat)
        betax = self.tmat @ betap
        beta = np.roll(betax, i-1, axis=0)
        return beta

    def invtransform(self, b1, b2, b3):
        bary = np.array([b1,b2,b3])
        i = (Barycentric.segment(bary) ) % 3
        betax = np.roll(bary, -i, axis=0)
        betap = self.tmatinv @ betax
        pj = self.subproj[i]#FIXME ?
        return pj.invtransform(*betap)

class SnyderEASym(BarycentricMapProjection):

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
            try:
                beta += b
            except NameError:
                beta = b
        return beta/3

    def invtransform(self, *args, **kwargs):
        return NotImplemented

def schwarz_fp(alpha, beta, gam):
    """Parameters of the Schwarz triangle map.
    Args:
        alpha, beta, gamma: Equal to pi times an angle of the triangle.
    Returns:
        s1: Value of the Schwarz triangle map at z=1.
        sinf: Value of the Schwarz triangle map at z=infinity.
        scale: Scale factor for spherical triangles. Will be zero or undefined
        if alpha + beta + gamma <= 1.
    """
    a = (1 - alpha - beta - gam)/2
    b = (1 - alpha + beta - gam)/2
    c = 1 - alpha
    palpha = np.pi*alpha
    pbeta = np.pi*beta
    pgam = np.pi*gam
    gfact = gamma(2-c)/(gamma(1-a)*gamma(c))
    s1 = gamma(c-a)*gamma(c-b)/gamma(1-b)*gfact
    sinf = np.exp(1j*palpha)*gamma(b)*gamma(c-a)*gfact/gamma(b-c+1)
    scale = sqrt(abs((np.cos(palpha+pbeta)+np.cos(pgam))/
                 (np.cos(palpha-pbeta)+np.cos(pgam))))
    return s1, sinf, scale

def c2c_mobius_finite(z,zi,wi):
    """Mobius transformation defined by mapping the points in zi to the points
    in wi."""
    ones = np.ones(zi.shape)
    a = np.linalg.det(np.stack([zi*wi,wi,ones]))
    b = np.linalg.det(np.stack([zi*wi,zi,wi]))
    c = np.linalg.det(np.stack([zi,wi,ones]))
    d = np.linalg.det(np.stack([zi*wi,zi,ones]))
    return (a*z+b)/(c*z+d)

def c2c_mobius_01inf(z, z0=0, z1=1, zinf=1j ):
    """Mobius transformation defined by mapping 3 points to 0, 1, infinity"""
    if ~np.isfinite(zinf):
        return (z-z0)/(z1-z0)
    elif ~np.isfinite(z1):
        return (z-z0)/(z-zinf)
    elif ~np.isfinite(z0):
        return (z1-zinf)/(z-zinf)
    else:
        return (z-z0)*(z1-zinf)/((z-zinf)*(z1-z0))

class ConformalTri(CtrlPtsProjection):
    nctrlpts = 3

    def __init__(self, ctrlpts, tgtpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod=geod)
        self.tgtpts = float2d_to_complex(tgtpts.T).squeeze()

        actrlpts = ctrlpts
        basei = 0
        basept = actrlpts[:, basei]
        crsst = {'proj': 'stere',
                 'lon_0': basept[0],
                 'lat_0': basept[1]}
        world_crs = {'init': 'epsg:4326'}
        stert = pyproj.transformer.Transformer.from_crs(world_crs,
                                                             crs_to=crsst)
        sterti = pyproj.transformer.Transformer.from_crs(crsst,
                                                         crs_to=world_crs)
        self.stert = stert
        self.sterti = sterti
        self.ctrl_s1, self.ctrl_sinf, self.ctrl_scale = schwarz_fp(*self.ctrl_angles/180)

        alpha, beta, gam = self.ctrl_angles/180
        self.a = (1 - alpha - beta - gam)/2
        self.b = (1 - alpha + beta - gam)/2
        self.c = 1 - alpha
        self.ap = (1 + alpha - beta - gam)/2#a - c + 1
        self.bp = (1 + alpha + beta - gam)/2#b - c + 1
        self.cp = 1 + alpha#2-c

        tgt_sides = abs(np.roll(self.tgtpts, 1) - np.roll(self.tgtpts, -1))
        tgt_angles = anglesgivensides(tgt_sides, scale=1)[0]
        alphat, betat, gamt = tgt_angles/np.pi
        self.apt = (1 + alphat - betat - gamt)/2
        self.bpt = (1 + alphat + betat - gamt)/2#
        self.cpt = 1 + alphat
        self.ct = 1 - alphat

        self.t1_s1, self.t1_sinf, _ = schwarz_fp(alphat, betat, gamt)

        self.pts_t = np.array(stert.transform(actrlpts[0], actrlpts[1]))
        self.pts_c = float2d_to_complex(self.pts_t.T.copy()).squeeze()
        #pts_r = pts_c / pts_c[1] * ctrl_s1
        self.bx = self.tgtpts[0]
        self.ax = (self.tgtpts[1] - self.tgtpts[0])/self.t1_s1

    def transform(self, lon, lat):
        lon + 0
        testpt_t = np.array(self.stert.transform(lon, lat))
        testpt_c = float2d_to_complex(testpt_t).squeeze()
        testpt_r = testpt_c / self.pts_c[1] * self.ctrl_s1

        a = self.a
        b = self.b
        c = self.c
        ap = self.ap
        bp = self.bp
        cp = self.cp

        def objective(t):
            z = t.view(dtype=complex)
            result = z**(1-c)*hyp2f1(ap,bp,cp,z)/hyp2f1(a,b,c,z)
            return abs(result - testpt_r)
        initial = c2c_mobius_01inf(testpt_r,
                                   z1=self.ctrl_s1, zinf=self.ctrl_sinf)
        res = minimize(objective, x0=[initial.real, initial.imag],
                       method='Nelder-Mead', options={'maxiter': 1E3})
        h = res.x.view(dtype=np.complex)

        ct = self.ct
        apt = self.apt
        bpt = self.bpt
        cpt = self.cpt
        testpt_t1 = h**(1-ct)*hyp2f1(apt,bpt,cpt,h)
        final = self.ax*testpt_t1 + self.bx
        return complex_to_float2d(final).T

    def invtransform(self, x, y):
        final = x + 1j*y
        testpt_t1i = (final - self.bx)/self.ax
        ct = self.ct
        apt = self.apt
        bpt = self.bpt
        cpt = self.cpt
        a = self.a
        b = self.b
        c = self.c
        ap = self.ap
        bp = self.bp
        cp = self.cp
        def objectivei(t):
            z = t.view(dtype=complex)
            result = z**(1-ct)*hyp2f1(apt,bpt,cpt,z)
            return abs(result - testpt_t1i)
        initiali = c2c_mobius_01inf(testpt_t1i,
                                    z1=self.t1_s1, zinf=self.t1_sinf)
        resi = minimize(objectivei, x0=[initiali.real, initiali.imag],
                       method='Nelder-Mead', options={'maxiter': 1E3})
        hi = resi.x.view(dtype=np.complex)

        testpt_ri = hi**(1-c)*hyp2f1(ap,bp,cp,hi)/hyp2f1(a,b,c,hi)
        testpt_ci = testpt_ri * self.pts_c[1]/self.ctrl_s1
        testpt_ti = complex_to_float2d(testpt_ci).T
        testpt_i = self.sterti.transform(*testpt_ti)
        return testpt_i

class ConformalTri3(CtrlPtsProjection):
    nctrlpts = 3

    def __init__(self, ctrlpts, tgtpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod=geod)
        self.tgtpts = float2d_to_complex(tgtpts.T).squeeze()
        subproj = []
        for i in range(3):
            rc = np.roll(ctrlpts, -i, axis=1)
            rt = np.roll(self.tgtpts, -i)
            subproj.append(ConformalTri(rc, rt, geod=geod))
        self.subproj = subproj
        self.bp = Barycentric(tgtpts)

    def transform(self, lon, lat):
        i = self.lune(lon, lat)
        mp = self.subproj[i]
        return mp.transform(lon, lat)

    def segment(self, x, y):
        bp = self.bp
        bary = bp.invtransform(x, y)
        return bp.segment(bary)

    def invtransform(self, x, y):
        i = self.segment(x, y)
        sp = self.subproj[i]
        return sp.invtransform(x, y)

class Double(CtrlPtsProjection):
    def __init__(self, ctrlpts, proj1, proj2, t=0.5):
        subproj = [proj1(ctrlpts), proj2(ctrlpts)]
        self.nctrlpts = subproj[0].nctrlpts
        if self.nctrlpts != subproj[1].nctrlpts:
            raise ValueError('proj1 and proj2 have different # of ctrlpts')
        super().__init__(ctrlpts)
        self.subproj = subproj
        self.t = t

    def transform(self, lon, lat):
        subproj = self.subproj
        t = self.t
        return ((1 - t)*subproj[0].transform(lon, lat)
                + t*subproj[1].transform(lon, lat))

#%% quad
class CriderEq(UVMapProjection):
    def transform_v(self, ll):
        vtestpt = UnitVector.transform(*(ll.reshape(2,-1)))
        ctrlpts_v = self.ctrlpts_v
        result = []
        for p in [(0, 1, 2, 3),(1, 2, 3, 0)]:
            #FIXME can calculate a lot of this stuff beforehand
            v0 = ctrlpts_v[..., p[0]]
            v1 = ctrlpts_v[..., p[1]]
            v2 = ctrlpts_v[..., p[2]]
            v3 = ctrlpts_v[..., p[3]]
            cosw01 = v0 @ v1
            cosw23 = v2 @ v3
            w01 = np.arccos(cosw01)
            w23 = np.arccos(cosw23)
            #sinw01 = sqrt(1 - cosw01**2)
            #sinw23 = sqrt(1 - cosw23**2)
            w = (w01 + w23) / 2
            sinw = np.sin(w)
            cosw = np.cos(w)

            #vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = np.tensordot(vtestpt, np.cross(v1, v2), axes=(0,0))
            #vt23 = vtestpt @ np.cross(v2, v3)
            vt30 = np.tensordot(vtestpt, np.cross(v3, v0), axes=(0,0))
            vt02 = np.tensordot(vtestpt, np.cross(v0, v2), axes=(0,0))
            vt13 = np.tensordot(vtestpt, np.cross(v1, v3), axes=(0,0))
            a = vt12 - cosw * (vt02 + vt13) - vt30 * cosw**2
            b = sinw * (2 * vt30 * cosw + vt02 + vt13)
            c = -vt30 * sinw**2
            desc = b**2 - 4*a*c
            index = a != 0
            nump = np.where(index, -b + sqrt(desc), -c)
            denom= np.where(index, 2*a, b)
            j = np.arctan2(nump,denom)/w
            result.append(j)
        result = np.array(result)
        return result.reshape(ll.shape)

    def invtransform_v(self, pts):
        u = pts[0].flatten()[np.newaxis]
        v = pts[1].flatten()[np.newaxis]
        a = self.ctrlpts_v[..., 0, np.newaxis]
        b = self.ctrlpts_v[..., 1, np.newaxis]
        c = self.ctrlpts_v[..., 2, np.newaxis]
        d = self.ctrlpts_v[..., 3, np.newaxis]
        f = slerp(a,b,u)
        g = slerp(d,c,u)
        h = slerp(b,c,v)
        k = slerp(a,d,v)
        inv = np.cross(np.cross(f, g, axis=0),
                       np.cross(h, k, axis=0), axis=0)
        result = UnitVector.invtransform_v(inv)
        return result.reshape(pts.shape)

class Crider(UVMapProjection):

    def transform(self, lon, lat):
        vtestpt = UnitVector.transform(lon, lat)
        ctrlpts_v = self.ctrlpts_v
        lon + 0#will TypeError if lon is not a number
        result = []
        for p in [(0, 1, 2, 3),(1, 2, 3, 0)]:
            #FIXME can calculate a lot of this stuff beforehand
            v0 = ctrlpts_v[..., p[0]]
            v1 = ctrlpts_v[..., p[1]]
            v2 = ctrlpts_v[..., p[2]]
            v3 = ctrlpts_v[..., p[3]]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt23 = vtestpt @ np.cross(v2, v3)
            vt30 = vtestpt @ np.cross(v3, v0)
            vt02 = vtestpt @ np.cross(v0, v2)
            vt13 = vtestpt @ np.cross(v1, v3)
            cosw01 = v0 @ v1
            cosw23 = v2 @ v3
            w01 = np.arccos(cosw01)
            w23 = np.arccos(cosw23)
            sinw01 = sqrt(1 - cosw01**2)
            sinw23 = sqrt(1 - cosw23**2)
            if np.isclose(w01, w23):
                w = (w01 + w23) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                j = self._b_eq(w, sinw, cosw, vt01, vt02,
                               vt12, vt13, vt23, vt30,
                               v0, v1, v2, v3)
            else:
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
            #FIXME need a limit here to prevent results running away
            #if t < -lim or t > lim:
            #    return t**2, 2*t
            z = c1*np.cos(wm*t - d1) + c2*np.cos(wp*t - d2)
            dz = -c1*wm*np.sin(wm*t - d1) - c2*wp*np.sin(wp*t - d2)
            return z, dz

        res = root_scalar(objective, fprime=True, method='newton', x0=0.5)
        return res.root

    def _b_eq(self, w, sinw, cosw, vt01, vt02, vt12, vt13, vt23, vt30,
              v0,v1,v2,v3):
        a = vt12 - cosw * (vt02 + vt13) - vt30 * cosw**2
        b = sinw * (2 * vt30 * cosw + vt02 + vt13)
        c = -vt30 * sinw**2
        if a == 0:
            num = -c
            denom = b
            j = np.arctan2(num,denom)/w
            return j
        else:
            desc = b**2 - 4*a*c
            nump = -b + sqrt(desc)
            denom= 2*a
            jp = np.arctan2(nump,denom)/w
            return jp

    def invtransform(self, u, v):
        a = self.ctrlpts_v[..., 0]
        b = self.ctrlpts_v[..., 1]
        c = self.ctrlpts_v[..., 2]
        d = self.ctrlpts_v[..., 3]
        f = slerp(a,b,u)
        g = slerp(d,c,u)
        h = slerp(b,c,v)
        k = slerp(a,d,v)
        inv = np.cross(np.cross(f, g), np.cross(h, k))
        return UnitVector.invtransform_v(inv)

class SnyderEA4(CtrlPtsProjection):
    def __init__(self, ctrlpts, tgtpts=TGTPTS4):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        if ctrlpts.shape[1] != tgtpts.shape[1]:
            raise ValueError('ctrlpts and tgtpts have different lengths')
        nctrlpts = ctrlpts.shape[1]
        self.nctrlpts = nctrlpts
        self.tgtpts = tgtpts
        super().__init__(ctrlpts)
        center = self.center
        bcenter = tgtpts.mean(axis=1)
        self.bcenter = bcenter
        self.btargets = [np.concatenate([bcenter[:, np.newaxis],
                                         np.roll(TGTPTS4, -i, axis=1)[:, :2]],
                                        axis=1) for i in range(nctrlpts)]
        subproj = []
        bprojs = []
        arr = np.arange(nctrlpts)
        for i in range(nctrlpts):
            index = np.roll(arr, -i)[:2]
            cp = np.concatenate([center[:,np.newaxis],
                                 ctrlpts[:, index]], axis=1)
            pj = SnyderEA(cp)
            subproj.append(pj)
            bprojs.append(Barycentric(self.btargets[i]))
        self.subproj = subproj
        self.bprojs = bprojs

        #for segment
        bc1 = np.concatenate([bcenter, [1]], axis=0)
        tgt1 = np.concatenate([tgtpts, np.ones((1,tgtpts.shape[1]))], axis=0)
        bcxtgt = -np.cross(tgt1, bc1, axis=0)
        self.bcxtgt = bcxtgt

    def transform(self, lon, lat):
        subproj = self.subproj
        bprojs = self.bprojs
        i = self.lune(lon, lat)
        pj = subproj[i]#FIXME right offset?
        bp = bprojs[i]#FIXME same
        betap = pj.transform(lon, lat)
        uvp = bp.transform(*betap)
        return uvp

    def segment(self, u, v):

        bcxtgt = self.bcxtgt
        try:
            fill = np.ones(u.shape)
        except AttributeError:
            fill = 1
        uv1 = np.stack([u,v,fill], axis=0)
        #print(bcxtgt)
        #print(uv1)
        sk = bcxtgt.T @ uv1
        sg = sk >= 0
        ind = sg & ~np.roll(sg, shift=-1, axis=0)
        result = np.argmax(ind, axis=0)
        return result#.reshape(u.shape)

    def invtransform(self, u, v):
        u + 0
        i = self.segment(u, v)
        pj = self.subproj[i]#FIXME
        bp = self.bprojs[i]
        bary = bp.invtransform(u, v)
        return pj.invtransform(*bary)

#%% inverse-only ones
class KProjection(CtrlPtsProjection):
    exact = True
    k = 1
    def extend(self, v):
        normal = self.center_v
        k = self.k
        n = np.linalg.norm(v, axis=0, keepdims=True)
        if self.exact:
            vdotc = np.tensordot(v, normal, axes=(0, 0))[np.newaxis]
            vdotv = n**2
            p = -vdotc + sqrt(1 + vdotc**2 - vdotv)
        else:
            p = 1 - n
        #print(v.shape, p.shape, normal.shape)
        return v + k*p*normal[..., np.newaxis]

class ReverseFuller(BarycentricMapProjection):

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.tweak = tweak

    def transform(self, *args, **kwargs):
        return NotImplemented

    def invtransform(self, b1, b2, b3):
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        v01 = slerp(v1, v0, b1)
        v02 = slerp(v2, v0, b1)
        cx12 = normalize(np.cross(v01, v02))
        v12 = slerp(v2, v1, b2)
        v10 = slerp(v0, v1, b2)
        cx20 = normalize(np.cross(v12, v10))
        v20 = slerp(v0, v2, b3)
        v21 = slerp(v1, v2, b3)
        cx01 = normalize(np.cross(v20, v21))

        v0x = np.cross(cx20, cx01)
        v1x = np.cross(cx01, cx12)
        v2x = np.cross(cx12, cx20)
        vx = np.stack([v0x,v1x,v2x], axis=-1)
        if not self.tweak:
            vx = normalize(vx)
        result = vx.mean(axis=-1)
        result = self._fix_corners_inv(bi, result)
        return UnitVector.invtransform_v(result)

class NSlerpTri(BarycentricMapProjection, KProjection):

    def __init__(self, ctrlpts, k=1, exact=True, pow=1, eps=0):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        self.pow = pow
        angles = self.sides
        self.eq = (np.max(angles) - np.min(angles)) <= eps
        if self.eq:
            self.avangle = np.mean(angles)

    def transform(self, *args, **kwargs):
        return NotImplemented

    def _tri_naive_slerp_angles(self, bary):
        """Interpolates the angle factor so that it's equal to the
        angle between pts 1 and 2 when beta_3=0, etc.
        """
        angles = self.sides
        if self.eq:
            return self.avangle
        a = bary[0]
        b = bary[1]
        c = bary[2]
        ab = (a*b)**self.pow
        bc = (b*c)**self.pow
        ca = (c*a)**self.pow
        denom = ab + bc + ca
        numer = ab*angles[0] + bc*angles[1] + ca*angles[2]
        return numer/denom

    def invtransform_v(self, bary):
        base = self.ctrlpts_v
        angles = self._tri_naive_slerp_angles(bary)
        b = np.sin(angles * bary) / np.sin(angles)
        result = (b.T.dot(base.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(bary, result)
        return UnitVector.invtransform_v(result)

class NSlerpQuad(UVMapProjection, KProjection):

    def __init__(self, ctrlpts, k=1, exact=True, pow=1, eps=0):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        self.pow = pow
        angles = self.sides
        self.eq = (np.max(angles) - np.min(angles)) <= eps
        if self.eq:
            self.avangle = np.mean(angles)

    def transform(self, *args, **kwargs):
        return NotImplemented

    def _angles_interp(self, x, y):
        """Interpolates the angle factors separately so that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        pow= self.pow
        angles = self.sides
        ax = angles[0]
        bx = angles[2]
        ay = angles[3]
        by = angles[1]
        result1 = (ax*(1-y)**pow + bx*(y)**pow)/((1-y)**pow + (y)**pow)
        result2 = (ay*(1-x)**pow + by*(x)**pow)/((1-x)**pow + (x)**pow)
        return result1, result2

    def invtransform_v(self, v):
        """
        Naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        anglex, angley = self._angles_interp(x, y)
        sx = np.sin((x)*anglex)
        sy = np.sin((y)*angley)
        scx = np.sin((1-x)*anglex)
        scy = np.sin((1-y)*angley)
        a = scx * scy
        b = sx * scy
        c = sx * sy
        d = scx * sy
        mat = (np.stack([a, b, c, d], axis=-1) /
            (np.sin(anglex)* np.sin(angley))[..., np.newaxis] )
        result = (mat.dot(self.ctrlpts_v.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(x, y, result)
        return UnitVector.invtransform_v(result)

class NSlerpQuad2(UVMapProjection, KProjection):

    def __init__(self, ctrlpts, k=1, exact = True, pow=1, eps=0):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        self.pow = pow
        angles = self.sides
        self.eq = (np.max(angles) - np.min(angles)) <= eps
        if self.eq:
            self.avangle = np.mean(angles)

    def transform(self, *args, **kwargs):
        return NotImplemented

    def _angles_interp(self, x, y):
        """Interpolates the angle factor together that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        pow = self.pow
        angles = self.sides
        if self.eq:
            return self.avangle
        a = ((1-x)*(1-y)*x)**pow
        b = ((1-y)*x*y)**pow
        c = ((1-x)*x*y)**pow
        d = ((1-x)*(1-y)*y)**pow
        numer = a*angles[0] + b*angles[1] + c*angles[2] + d*angles[3]
        denom = a + b + c + d
        return numer/denom

    def invtransform_v(self, v):
        """
        Variant naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        angle = self._angles_interp(x, y)[..., np.newaxis]
        a = (1-x)*(1-y)
        b = x*(1-y)
        c = x*y
        d = (1-x)*y
        mat = (np.sin(np.stack([a, b, c, d], axis=-1)*angle) /
               np.sin(angle))
        result = (mat.dot(self.ctrlpts_v.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(x, y, result)
        return UnitVector.invtransform_v(result)

class EllipticalQuad(UVMapProjection, KProjection):
    """An extension of the elliptical map.
    """

    def __init__(self, ctrlpts, k=1, exact=True, eps=1E-6):
        """Parameters:
        ctrlpts: 2x4 Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        sidelength = self.sides
        assert abs(sidelength[0] - sidelength[2]) < eps
        assert abs(sidelength[1] - sidelength[3]) < eps
        vertangles = (np.roll(self.baz, -1) - self.faz) % 360
        assert abs((vertangles - vertangles.mean()).sum()) < eps
        ctrlpts_v = self.ctrlpts_v
        center_v = self.center_v
        midpoint_x = ctrlpts_v[:, 1] + ctrlpts_v[:, 2]
        midpoint_y = ctrlpts_v[:, 0] + ctrlpts_v[:, 1]
        m2 = np.cross(center_v, midpoint_y)
        m3 = np.cross(center_v, midpoint_x)
        mat = np.array([m2/np.linalg.norm(m2),
                        m3/np.linalg.norm(m3),
                        center_v]).T
        self.mat = mat
        self.invmat = np.linalg.inv(mat)
        self.rotctrlpts_v = self.invmat @ ctrlpts_v

    def transform(self, *args, **kwargs):
        return NotImplemented

    def invtransform_v(self, uv):
        #FIXME needs rotations
        rot_base = self.rotctrlpts_v
        a = rot_base[0,2]
        b = rot_base[1,2]
        c = rot_base[2,2]
        x = uv[0]*2 - 1
        y = uv[1]*2 - 1
        axt = (1 - a**2*x**2)
        byt = (1 - b**2*y**2)
        at = (1-a**2)
        bt = (1-b**2)
        u = a * x * sqrt(byt/bt)
        v = b * y * sqrt(axt/at)
        w = c * sqrt(axt*byt/(at*bt))
        result = np.stack([u,v,w], axis=0)
        result = self.mat @ result
        #print(v, result)
        result = self.extend(result)
        result = self._fix_corners_inv(uv[0], uv[1], result)
        return UnitVector.invtransform_v(result)

#%%
if __name__ == "__main__":
    import doctest
    sup = np.testing.suppress_warnings()
    sup.filter(RuntimeWarning)
    options = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    with sup:
        doctest.testmod(optionflags = options)
