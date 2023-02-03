# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:18:57 2023

@author: brsr
"""

import geopandas
import pandas as pd
import shapely
from shapely.geometry import LineString, Point #Polygon, 
import numpy as np

def sqrt(x):
    """Real sqrt clipped to 0 for negative values.

    >>> x = np.array([-np.inf, -1, 0, 1, np.inf, np.nan])
    >>> sqrt(x)
    array([ 0.,  0.,  0.,  1., inf, nan])
    """
    return np.where(x < 0, 0, np.sqrt(x))

#%% Conversions to and from arrays
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

#%% Plane geometry
def shoelace(pts):
    """Find area of polygon in the plane defined by pts, where pts is an
    array with shape (2,n).
    >>> pts = np.arange(6).reshape(2,-1)%4
    >>> shoelace(pts)
    2.0
    """
    return abs(np.sum(np.cross(pts, np.roll(pts, -1, axis=1), axis=0)))/2

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
#%% Spherical geometry with vectors
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

def rodrigues(center, v, theta):
    """Rodrigues formula: rotate vector v around center by angle theta
    """
    cxv = np.cross(center, v)
    cv = np.sum(center* v, axis=-1, keepdims=True)
    cc = v*np.cos(theta) + cxv*np.sin(theta) + center*cv*(1-np.cos(theta))
    return cc

#%% Geodesy-related functions
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

def antipode_v(ll):
    """Antipodes of points given by longitude and latitude."""
    antipode = ll.copy()
    antipode[0] -= 180
    index = antipode[0] < -180
    antipode[0, index] += 360
    antipode[1] *= -1
    return antipode

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
