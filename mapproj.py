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
#import scipy as sp
from scipy.optimize import minimize, minimize_scalar, root_scalar

#TODO:
#fix fuller tri - actually based on the paper
#implement snyder equal area (paper)
#implement conformal (some kind of circle-packing thing?)
#script comparisons

def ptseriestoarray(ser):
    return np.concatenate([x.xy for x in ser], axis=-1)

def arraytoptseries(arr):
    return geopandas.GeoSeries([Point(x[0], x[1])
                    for x in arr.reshape(2, -1).T])

def sqrt(x):
    """Real sqrt clipped to 0 for negative values"""
    return np.where(x < 0, 0, np.sqrt(x))

#arange3 = np.arange(3)

def fixbary_normalize(bary):
    bary = np.array(bary)
    return bary / np.sum(bary, axis=0, keepdims=True)

def fixbary_subtract(bary):
    bary = np.array(bary)
    s = np.sum(bary, axis=0, keepdims=True) - 1
    return bary - s/3

#FIRST AXIS IS SPATIAL

def transeach(func, geoms):
    """Transform each element of geoms using the function func."""
    plist = []
    for geom in geoms:
        plist.append(shapely.ops.transform(func, geom))
    return geopandas.GeoSeries(plist)

def latlontovector(lon, lat, scale=np.pi/180):
    """Convert latitude and longitude to 3-vector"""
    x = np.cos(lat*scale)*np.cos(lon*scale)
    y = np.cos(lat*scale)*np.sin(lon*scale)
    z = np.sin(lat*scale)

    return np.stack([x, y, z], axis=0)

def vectortolatlon(vector, scale=180/np.pi):
    """Convert 3-vector to latitude and longitude"""
    #vector = vector/np.linalg.norm(vector, axis=0, keepdims=True)
    lat = scale*np.arctan2(vector[2], np.sqrt(vector[1]**2 + vector[0]**2))
    lon = scale*np.arctan2(vector[1], vector[0])
    return np.stack([lon, lat])

def graticule(spacing1=15, spacing2=1,
              lonrange = [-180, 180], latrange = [-90, 90]):
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
    """only uses the first 2 angles"""
    angles = angles * scale
    p0 = [np.cos(2*angles[1]), np.sin(2*angles[1])]
    p1 = [np.cos(2*angles[0]), np.sin(-2*angles[0])]
    p2 = [1, 0]
    return np.array([p0, p1, p2]).T

def trigivenlengths(sidelengths):
    r = np.product(sidelengths)/sqrt(
            2*np.sum(sidelengths**2*np.roll(sidelengths,1)**2)
            -np.sum(sidelengths**4))
    angles = np.arcsin(sidelengths/r/2)
    return r*trigivenangles(np.roll(angles, -1), scale=1)
#%%
def authalic_lat(phi, geod, scale = np.pi/180):
    """Convert geodetic latitude to authalic
    """
    phi = phi * scale
    e   = np.sqrt(geod.es)
    qp  = 1 + (1 - e**2)/e * np.arctanh(e)
    q   = ((1 - e**2)*np.sin(phi)/(1 - (e*np.sin(phi))**2) +
           (1 - e**2)/e * np.arctanh(e*np.sin(phi)) )
    return np.arcsin(q/qp)/scale

def gerp(v1, v2, t, geod):
    """Geodetic interpolation"""
    f, b, r = geod.inv(v1[0], v1[1], v2[0], v2[1])
    lon, lat, b = geod.fwd(v1[0], v1[1], f, r*t)
    return lon, lat

def invgerp(v1, v2, vt, geod):
    """Inverse geodetic interpolation"""
    f, b, r = geod.inv(v1[0], v1[1], v2[0], v2[1])
    f, b, rt = geod.inv(v1[0], v1[1], vt[0], vt[1])
    return rt/r

def gintersect(a0, a1, b0, b1, geod, initial=None):
    """Find point of intersection of geodesic from a0 to a1 and
    from b0 to b1
    """
    def objective(t):
        pt = gerp(a0, a1, t, geod=geod)
        area, _ = geod.polygon_area_perimeter([b0[0], b1[0], pt[0]],
                                              [b0[1], b1[1], pt[1]])
        return area
    if initial is None:
        x0 = 1
    else:
        x0 = invgerp(a0, a1, initial, geod)
    x1 = 1 if abs(x0) < 0.5 else 0
    res = root_scalar(objective, x0=x0, x1=x1)
    if not res.converged:
        print(res)
    return gerp(a0, a1, res.root, geod=geod)

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
    array([[ 1.       ,  0.       ,  0.       ],
           [ 0.8660254,  0.       ,  0.5      ],
           [ 0.5      ,  0.       ,  0.8660254],
           [ 0.       ,  0.       ,  1.       ]])
    """
    t = intervals
    angle = central_angle(pt1, pt2)[..., np.newaxis]
    return (np.sin((1 - t)*angle)*pt1 + np.sin((t)*angle)*pt2)/np.sin(angle)

def shoelace(pts):
    return abs(np.sum(np.cross(pts, np.roll(pts, -1, axis=1), axis=0)))/2
    
def omegascale(adegpts, degpts_t, actrlpts, tgtpts, geod):
    ar, p = geod.polygon_area_perimeter(actrlpts[0], actrlpts[1])
    at = shoelace(tgtpts)
    es = geod.es
    a = geod.a
    factor = np.pi/180
    lon = adegpts[0]*factor
    lat = adegpts[1]*factor
    x = degpts_t[0]
    y = degpts_t[1]
    dx = np.gradient(x, factor, edge_order=2)
    dy = np.gradient(y, factor, edge_order=2)
    dxdlat, dxdlon = dx
    dydlat, dydlon = dy
    R = a*np.sqrt(1-es)/(1-es*np.sin(lat)**2)
    h = sqrt((dxdlat)**2 + (dydlat)**2)*(1-es*np.sin(lat)**2)**(3/2)/(a*(1-es))
    k = sqrt((dxdlon)**2 + (dydlon)**2)*(1-es*np.sin(lat)**2)**(1/2)/(a*np.cos(lat))
    sinthetaprime = np.clip(((dydlat*dxdlon - dxdlat*dydlon)/(R**2*h*k*np.cos(lat))),
                            -1, 1)
    aprime = sqrt(h**2 + k**2 + 2*h*k*sinthetaprime)
    bprime = sqrt(h**2 + k**2 - 2*h*k*sinthetaprime)
    sinomegav2 = np.clip(bprime/aprime, -1, 1)
    omega = 360*np.arcsin(sinomegav2)/np.pi
    scale = h*k*sinthetaprime*at/ar
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
    _bilinear_mat = np.array([[ 1, 1, 1, 1],
                              [-1, 1, 1,-1],
                              [-1,-1, 1, 1],
                              [ 1,-1, 1,-1]])/4
    def __init__(self, tgtpts):
        self.tgtpts = tgtpts
        self.abcd = self._bilinear_mat @ tgtpts.T

    def transform(self, u, v):
        """Bilinear interpolation
        u and v should have the same shape"""
        abcd = self.abcd
        stack = np.stack([np.ones(u.shape), u, v, u*v])
        return (abcd @ stack).T

    def transform_v(self, pts, **kwargs):
        return self.transform(pts[0], pts[1])

    def invtransform_v(self, pts):
        """Bilinear interpolation"""
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

class Homomorphism(Projection):
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
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        self.geod = geod
        self.ctrlpts = ctrlpts
        self.ctrlvector = latlontovector(ctrlpts[0], ctrlpts[1])
        faz, baz, sidelengths = self.geod.inv(ctrlpts[0], ctrlpts[1],
                                          np.roll(ctrlpts[0], -1),
                                          np.roll(ctrlpts[1], -1))
        area, _ = geod.polygon_area_perimeter(ctrlpts[0], ctrlpts[1])
        self.sidelengths = sidelengths
        self.faz = faz
        self.baz = baz
        self.ctrlarea = area

    def orienttgtpts(self, tgtpts, N = (0, 90)):
        """Orient target points so that line from 0 to the projection of N
        points up. Will fail if map projection doesn't use tgtpts."""
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

    def setmat(self):
        """Set matrices that use tgtpts"""
        tgtpts = self.tgtpts
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
        return vectortolatlon(vector).reshape(pts.shape)

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
        vctrl = latlontovector(ctrlpts[0], ctrlpts[1])
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
        result = vectortolatlon(vector).reshape(shape)
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

    def _fix_corners_uv(self, x, y, result):
        index0 = (x == -1) & (y == -1)
        index1 = (x == 1) & (y == -1)
        index2 = (x == 1) & (y == 1)
        index3 = (x == -1) & (y == 1)
        result[..., index0] = self.ctrlvector[..., 0, np.newaxis]
        result[..., index1] = self.ctrlvector[..., 1, np.newaxis]
        result[..., index2] = self.ctrlvector[..., 2, np.newaxis]
        result[..., index3] = self.ctrlvector[..., 3, np.newaxis]
        return result

    def _fix_corners_bary(self, bary, result):
        index0 = (bary[0] == 1)
        index1 = (bary[1] == 1)
        index2 = (bary[2] == 1)
        result[..., index0] = self.ctrlvector[..., 0, np.newaxis]
        result[..., index1] = self.ctrlvector[..., 1, np.newaxis]
        result[..., index2] = self.ctrlvector[..., 2, np.newaxis]
        return result
    
#%%
class FullerTriSph(SphereProjection):
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
        
    def transform(self, lat, lon):
        sidelengths = self.sidelengths
        vtestpt = latlontovector(lon, lat)
        vctrlpts = self.ctrlvector
        
        cosARC = np.mean(np.cos(sidelengths))
        sinARC = np.mean(np.sin(sidelengths))
        sij = vtestpt @ np.cross(np.roll(vctrlpts, 1, axis=1), 
                                 np.roll(vctrlpts, -1, axis=1), axis=0)
        a = np.arctan(sinARC*sij/
                      (sij*cosARC + np.roll(sij, 1) + np.roll(sij, -1)))
        ap = a/sidelengths
        return self.fixbary(ap)

    def invtransform(self, b1, b2, b3):
        return NotImplemented
#%%
class FullerQuadSph(SphereProjection):
    def transform(self, lat, lon):
        vtestpt = latlontovector(lat, lon)
        vctrlpts = self.ctrlvector
        result = []
        def det(*args):
            m = np.stack(args)
            return np.linalg.det(m)
        for p in [(1,3),(3,1)]:
            w01 = np.arccos(vctrlpts[..., 0] @ vctrlpts[..., p[0]])
            w23 = np.arccos(vctrlpts[..., 2] @ vctrlpts[..., p[1]])
            w = np.mean([w01, w23])
            p_01 = vctrlpts[..., 0] + vctrlpts[..., p[0]]
            p_32 = vctrlpts[..., p[1]] + vctrlpts[..., 2]
            m_01 = vctrlpts[..., 0] - vctrlpts[..., p[0]]
            m_32 = vctrlpts[..., p[1]] - vctrlpts[..., 2]
            v = vtestpt
            a = det(v, m_01, m_32)*(1 + np.cos(w))
            b = (det(v, p_01, m_32) + det(v, m_01, p_32))*np.sin(w)
            c = det(v, p_01, p_32)*(1 - np.cos(w))
            if a == 0:
                q = -c/b
            else:
                desc = b**2 - 4*a*c
                q = (-b + sqrt(desc))/(2*a)#probably this one
                qm = (-b - sqrt(desc))/(2*a)
                print(q, qm)
            j = np.arctan(q)*2/w
            result.append(j)
        return result

    # def invtransform_v(self, v):#
    #     """
    #     Naive slerp on a spherical quadrilateral.
    #     """
    #     x = v[0]
    #     y = v[1]
    #     angley = self.sidelengths[[0,2]].mean()
    #     anglex = self.sidelengths[[1,3]].mean()
    #     sx = np.sin((1+x)*anglex/2)
    #     sy = np.sin((1+y)*angley/2)
    #     scx = np.sin((1-x)*anglex/2)
    #     scy = np.sin((1-y)*angley/2)
    #     a = scx * scy
    #     b = sx * scy
    #     c = sx * sy
    #     d = scx * sy
    #     mat = (np.stack([a, b, c, d], axis=-1) /
    #         (np.sin(anglex)* np.sin(angley)) )
    #     result = (mat.dot(self.ctrlvector.T)).T
    #     #result = self._fix_corners(x, y, result)
    #     return vectortolatlon(result)

    def invtransform(self, u, v):
        a = self.ctrlvector[..., 0]
        b = self.ctrlvector[..., 1]
        c = self.ctrlvector[..., 2]
        d = self.ctrlvector[..., 3]
        x = (u + 1)/2
        y = (v + 1)/2
        f = slerp(a,b,x)
        g = slerp(d,c,x)
        h = slerp(b,c,y)
        k = slerp(a,d,y)
        inv = np.cross(np.cross(f, g), np.cross(h, k))
        return vectortolatlon(inv)
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
        result = self._fix_corners(bary, result)
        return vectortolatlon(result)

class NSlerpQuad(SphereProjection):
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
        return vectortolatlon(result)


class NSlerpQuad2(SphereProjection):
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
        return vectortolatlon(result)

class EllipticalQuad(SphereProjection):
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
        return vectortolatlon(result)
#%%
class FullerTri(MapProjection):
    def __init__(self, ctrlpts, geod, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts, geod)
        self.tweak = tweak
        if tweak:
            self.fixbary = fixbary_normalize

    def fixbary(self, bary):
        if self.tweak:
            return fixbary_normalize(bary)
        else:
            return fixbary_subtract(bary)

    def transform(self, lat, lon):
        try:
            result = []
            geod = self.geod
            for i in range(3):
                a = np.roll(self.ctrlpts, -i, axis=1)
                def objective(t):
                    pta = gerp(a[..., 1], a[..., 0], t, geod=geod)
                    ptb = gerp(a[..., 2], a[..., 0], t, geod=geod)
                    f, b, d = geod.inv(pta[0], pta[1], ptb[0], ptb[1])
                    f1, b1, d1 = geod.inv(pta[0], pta[1], lat, lon)
                    #obj = (f-f1) % 360 - 180
                    #obj = np.where(obj <= -180, obj + 180, obj)
                    obj = np.sin((f-f1)/180*np.pi)
                    #obj =  (((f-f1 - 90) % 180) - 90)**2
                    return obj
                try:
                    res = root_scalar(objective, bracket=[0, 1], method='brentq')#
                    #res = minimize_scalar(objective, bounds=[-1, 2])
                    result.append(res.root)
                except ValueError:
                    result.append(1E20)
            return tuple(self.fixbary(result))
        except pyproj.geod.GeodError:
            raise TypeError()
            
    def _transform_sph(self, lat, lon):
        vtestpt = latlontovector(lon, lat)
        vctrlpts = self.ctrlvector
        cosARC = np.mean(np.cos(self.sidelengths))
        sij = vtestpt @ np.cross(np.roll(vctrlpts, 1, axis=1), 
                                 np.roll(vctrlpts, -1, axis=1), axis=0)
        a = np.arctan(2*cosARC*sij/
                      (sij*cosARC + np.roll(sij, 1) + np.roll(sij, -1)))
        return a

    def invtransform(self, b1, b2, b3):
        bary = [b1, b2, b3]
        geod = self.geod
        actrlpts = self.ctrlpts
        linepts=[]
        for i in range(3):
            x = bary[i]
            ar = np.roll(actrlpts, -i, axis=1)
            a = ar[..., 0]
            b = ar[..., 1]
            c = ar[..., 2]
            f = gerp(b, a, x, geod=geod)
            g = gerp(c, a, x, geod=geod)
            linepts.append((f,g))

        intersectpts = []
        initial = self._invtransform_sph(linepts)
        if geod.sphere:
            result= initial
        else:
            for i in range(3):
                line1 = linepts[i]
                line2 = linepts[(i+1)%3]
                try:
                    x = gintersect(line1[0], line1[1], line2[0], line2[1], 
                                   geod=geod, initial=initial)
                except ZeroDivisionError:
                    x = initial
                intersectpts.append(x)
            result = np.mean(intersectpts, axis=0)
        if b1 == 1:
            result = actrlpts[..., 0]
        if b2 == 1:
            result = actrlpts[..., 1]
        if b3 == 1:
            result = actrlpts[..., 2]
        return result
    
    def _invtransform_sph(self, linepts):
        linepts = np.array(linepts)
        vpts = latlontovector(linepts[..., 0], linepts[..., 1])
        h = np.cross(vpts[...,0], vpts[...,1], axis=0)
        inv = np.cross(h, np.roll(h, -1, axis=-1), axis=0)
        if not self.tweak:
            inv = inv/np.linalg.norm(inv, axis=0, keepdims=True)  
        result = np.sum(inv, axis=1)    
        return vectortolatlon(result)   

#%%
class FullerQuad(MapProjection):
    def transform(self, lat, lon):
        geod = self.geod
        actrlpts = self.ctrlpts
        initial = self._transform_sph_rhomb(lat, lon)
        result = []
        flip = [(1,3),(3,1)]
        for i in range(2):
            p = flip[i]
            def objective(t):
                a = (t+1)/2
                pta = gerp(actrlpts[..., 0], actrlpts[..., p[0]],
                           a, geod=geod)
                ptb = gerp(actrlpts[..., p[1]], actrlpts[..., 2],
                           a, geod=geod)
                area, _ = geod.polygon_area_perimeter([pta[0], ptb[0], lat],
                                                      [pta[1], ptb[1], lon])
                return area
            x0 = initial[i]
            x1 = 1 if abs(initial[i]) < 0.1 else 0
            try:
                res = root_scalar(objective, x0=x0, x1=x1)#x0=initial[i])
                if not res.converged:
                    print(res)
                result.append(res.root)
            except ZeroDivisionError:
                result.append(1/2)
        return result

    def _transform_sph_rhomb(self, lat, lon):
        vtestpt = latlontovector(lat, lon)
        vctrlpts = self.ctrlvector
        result = []
        def det(*args):
            m = np.stack(args)
            return np.linalg.det(m)
        for p in [(1,3),(3,1)]:
            w01 = np.arccos(vctrlpts[..., 0] @ vctrlpts[..., p[0]])
            w23 = np.arccos(vctrlpts[..., 2] @ vctrlpts[..., p[1]])
            w = np.mean([w01, w23])
            p_01 = vctrlpts[..., 0] + vctrlpts[..., p[0]]
            p_32 = vctrlpts[..., p[1]] + vctrlpts[..., 2]
            m_01 = vctrlpts[..., 0] - vctrlpts[..., p[0]]
            m_32 = vctrlpts[..., p[1]] - vctrlpts[..., 2]
            v = vtestpt
            a = det(v, m_01, m_32)*(1 + np.cos(w))
            b = (det(v, p_01, m_32) + det(v, m_01, p_32))*np.sin(w)
            c = det(v, p_01, p_32)*(1 - np.cos(w))
            if a == 0:#not sure if a is ever non-zero
                q = -c/b
            else:
                desc = b**2 - 4*a*c
                q = (-b + sqrt(desc))/(2*a)#probably this one
                #qm = (-b - sqrt(desc))/(2*a)
            j = np.arctan(q)*2/w
            result.append(j)
        return result

    def invtransform(self, u, v):
        geod = self.geod
        initialpt = self._invtransform_sph(u, v)
        if geod.sphere:
            return initialpt
        actrlpts = self.ctrlpts
        #could switch this to enhance numerical stability or convergence,
        #doesn't seem to really matter
        pt01 = gerp(actrlpts[..., 0], actrlpts[..., 1], (u+1)/2, geod=geod)
        pt32 = gerp(actrlpts[..., 3], actrlpts[..., 2], (u+1)/2, geod=geod)
        pt03 = gerp(actrlpts[..., 0], actrlpts[..., 3], (v+1)/2, geod=geod)
        pt12 = gerp(actrlpts[..., 1], actrlpts[..., 2], (v+1)/2, geod=geod)
        return gintersect(pt01, pt32, pt03, pt12, geod=geod, initial=initialpt)
        def objective(t):
            pt = gerp(pt01, pt32, (t+1)/2, geod=geod)
            area, _ = geod.polygon_area_perimeter([pt03[0], pt12[0], pt[0]],
                                                  [pt03[1], pt12[1], pt[1]])
            return area
        x0 = invgerp(pt01, pt32, initialpt, geod)
        x1 = 1 if abs(x0) < 0.1 else 0
        res = root_scalar(objective, x0=x0, x1=x1)
        if not res.converged:
            print(res)
        invpt = gerp(pt01, pt32, (res.root+1)/2, geod=geod)
        return invpt

    def _invtransform_sph(self, u, v):
        a = self.ctrlvector[..., 0]
        b = self.ctrlvector[..., 1]
        c = self.ctrlvector[..., 2]
        d = self.ctrlvector[..., 3]
        x = (u + 1)/2
        y = (v + 1)/2
        f = slerp(a,b,x)
        g = slerp(d,c,x)
        h = slerp(b,c,y)
        k = slerp(a,d,y)
        inv = np.cross(np.cross(f, g), np.cross(h, k))
        return vectortolatlon(inv)
        
class ConformalTri(MapProjection):
    pass

class EqualAreaTri(MapProjection):
    pass
