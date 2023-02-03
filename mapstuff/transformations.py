# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:39:15 2023

@author: brsr
"""
import geopandas
#import pandas as pd
#import shapely
from shapely.geometry import LineString, Polygon#, Point
#import homography
import numpy as np
from abc import ABC

from .helper import graticule, sqrt
#%%
class Transformation(ABC):
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
class UV(Transformation):
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

class Barycentric(Transformation):
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

class UnitVector(Transformation):
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
