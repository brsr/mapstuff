#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:38:02 2019

@author: brsr
"""

import geopandas
import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib.pyplot as plt
import warnings
import numpy as np
#import scipy as sp
#from scipy.optimize import root_scalar#, minimize_scalar

#FIRST AXIS IS SPATIAL

def transeach(func, geoms):
    """Transform each element of geoms using the function func."""
    plist = []
    for geom in geoms:
        plist.append(shapely.ops.transform(func, geom))
    return geopandas.GeoSeries(plist)

def latlontovector(lon, lat,scale=np.pi/180):
    """Convert latitude and longitude to 3-vector"""
    x = np.cos(lat*scale)*np.cos(lon*scale)
    y = np.cos(lat*scale)*np.sin(lon*scale)
    z = np.sin(lat*scale)

    return np.stack([x,y,z], axis=0)

def vectortolatlon(vector, scale=180/np.pi):
    """Convert 3-vector to latitude and longitude"""
    #vector = vector/np.linalg.norm(vector, axis=0, keepdims=True)
    lat = scale*np.arctan2(vector[2], np.sqrt(vector[1]**2 + vector[0]**2))
    lon = scale*np.arctan2(vector[1], vector[0])
    return np.stack([lon, lat])

def barytoplane(bary, tgtpts):
    """Convert barycentric coordinates to plane coordinates with respect to
    tgtpts. tgtpts should be a 2x3 numpy array."""
    return tgtpts @ bary

def planetobary(xy, tgtpts):
    """Convert points in the 2d plane to barycentric coordinates with respect
    to tgtpts. tgtpts should be a 2x3 numpy array."""
    shape = list(xy.shape)
    shape[0] = 1
    xy1 = np.concatenate([xy, np.ones(shape)])
    m = np.concatenate([tgtpts, np.ones((1, 3))])
    return np.linalg.solve(m, xy1)
    
def graticule(spacing1=15, spacing2=1):
    plx = np.linspace(-180, 180, num=360/spacing2 + 1)
    ply = np.linspace(-90, 90, num=180/spacing1 + 1)[1:-1]
    mex = np.linspace(-180, 180, num=360/spacing1, endpoint=False)
    mey = np.linspace(-90, 90, num=180/spacing2 + 1)
    parallels = np.stack(np.meshgrid(plx, ply), axis=-1).transpose((1,0,2))
    meridians = np.stack(np.meshgrid(mex, mey), axis=-1)
    gratlist = [parallels[:, i] for i in range(parallels.shape[1])]
    gratlist += [meridians[:, i] for i in range(meridians.shape[1])]
    gratl2 = [LineString(line) for line in gratlist]
    grat = geopandas.GeoSeries(gratl2)
    grat.crs = {'init': 'epsg:4326'}
    return grat

class MapProjection():
    def __init__(self, ctrlpts, geoid):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geoid= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        self.geoid = geoid
        self.ctrlpts = ctrlpts
        self.settgtpts(ctrlpts)
        self.orienttgtpts(self.tgtpts)
    
    def settgtpts(self, ctrlpts):
        """Creates target triangle with edges the same length as the control 
        triangle, having its circumcenter at 0.
        Override this if you want a different triangle."""
        faz, baz, sidelengths = self.geoid.inv(ctrlpts[0], ctrlpts[1],
                                          np.roll(ctrlpts[0], -1),
                                          np.roll(ctrlpts[1], -1))
        x = (sidelengths[0]**2 - sidelengths[1]**2 +
             sidelengths[2]**2)/(2*sidelengths[0])
        y = np.sqrt(sidelengths[2]**2 - x**2)
        tgtpts = np.array([[0, sidelengths[0], x],
                   [0, 0, y]])
        boffset = np.roll(sidelengths, -1)**2 * (np.roll(sidelengths, 1)**2 +
                  sidelengths**2 - np.roll(sidelengths, -1)**2)
        boffset = boffset / np.sum(boffset)
        offset = tgtpts @ boffset
        tgtpts = tgtpts - offset[:, np.newaxis]
        self.tgtpts = tgtpts
        return tgtpts
    
    def orienttgtpts(self, tgtpts, N = (0, 90)):
        """Orient target points so that line from 0 to the projection of N 
        points up."""
        pN = self.transform(*N)
        #print(N, pN)
        if np.allclose(pN, [0,0]):
            raise ValueError('projection of N too close to 0')
        angle = np.arctan2(pN[0],pN[1])
        rotm = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
        #print(rotm.shape, tgtpts.shape)
        result = rotm @ tgtpts
        #print(result.shape)
        self.tgtpts = result
    
    def transform(self, x, y, **kwargs):
        pts = np.stack([x,y])
        vresult = self.transform_v(pts, **kwargs)
        return vresult[0], vresult[1]
        
    def invtransform(self, x, y, **kwargs):
        pts = np.stack([x,y])
        vresult = self.invtransform_v(pts, **kwargs)
        return vresult[0], vresult[1]
    
    def transform_v(self, *args, **kwargs):
        return NotImplemented

    def invtransform_v(self, *args, **kwargs):
        return NotImplemented

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

    def __init__(self, ctrlpts, geoid):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geoid= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        self.geoid = geoid
        self.ctrlpts = ctrlpts
        self.settgtpts(ctrlpts)
        self.radius = ((geoid.a**(3/2) + geoid.b**(3/2))/2)**(2/3)
        vctrl = latlontovector(ctrlpts[0], ctrlpts[1])
        self.ctrlvector = vctrl
        self.invctrlvector = np.linalg.pinv(vctrl)
        self.invperpmatrix = np.linalg.pinv(vctrl.T @ vctrl)
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
            radi = []
            for a, b in self.ctrlpts.T:
                radi.append(self.geoid.line_length([x,a],[y,b]))
            rad.append(radi)
        shape = list(pts.shape)
        shape[0] = 3
        rad = np.array(rad).T
        radsq = np.array(rad)**2
        result = self.m @ radsq
        return result.reshape(pts.shape)        
        
    def invtransform_v(self, pts, n=20, stop=1E-8):
        if geod.a != geod.b:
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
    def __init__(self, ctrlpts, geoid):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geoid: a pyproj.Geod object. For a unit sphere use
                pyproj.Geod(a=1,b=1).
        """
        self.geoid = geoid
        self.ctrlpts = ctrlpts
        #it's possible to get a geoid where this would give the wrong answer,
        #but it's not likely
        area, _ = geoid.polygon_area_perimeter([0,120,-120],[0,0,0])
        self.totalarea = 2*area
        self.ctrlarea, _ = geoid.polygon_area_perimeter(ctrlpts[0], 
                                                        ctrlpts[1])
        vctrl = latlontovector(ctrlpts[0], ctrlpts[1])
        self.ctrlvector = vctrl        
        a_i = np.sum(np.roll(self.ctrlvector, -1, axis=1) * 
                          np.roll(self.ctrlvector, 1, axis=1), axis=0)
        self.a_i = a_i
        self.b_i = (np.roll(a_i, -1) + np.roll(a_i, 1))/(1+a_i)
        self.tau_c = self.tau(self.ctrlarea)
        self.settgtpts(ctrlpts)
        try:
            self.orienttgtpts(self.tgtpts)
        except ValueError:
            pass

    def tau(self, area):
        """Convert areas on the geoid to tau values for inverse transform"""
        return np.tan(area/self.totalarea*2*np.pi)
    
    def transform(self, x, y):
        bary = self.transform_to_bary(x, y)
        return tuple(barytoplane(bary, self.tgtpts))
    
    def transform_to_bary(self, x, y):
        try:
            areas = []
            for i in range(3):
                smtri = self.ctrlpts.copy()
                smtri[:, i] = np.array([x,y])
                a, _ = self.geoid.polygon_area_perimeter(smtri[0], 
                                                         smtri[1])
                areas.append(a)
            areas = np.array(areas)
            return areas/self.ctrlarea
        except ValueError:
            raise TypeError()

    def transform_v(self, pts):
        bary = []
        rpts = pts.reshape((2, -1)).T   
        for x, y in rpts:
            bary.append(self.transform_to_bary(x,y))
        bary = np.array(bary).T
        plane = barytoplane(bary, self.tgtpts)
        shape = list(pts.shape)
        shape[0] = self.tgtpts.shape[0]
        return plane.reshape(shape)
    
    def invtransform_v(self, pts):
        if geod.a != geod.b:
            warnings.warn('inverse transform is approximate on ellipsoids')
        rpts = pts.reshape((2,-1))
        beta = planetobary(rpts, self.tgtpts)
        b_i = self.b_i[:,np.newaxis]
        tau = self.tau_c
        tau_i = self.tau(self.ctrlarea*beta)
        t_i = tau_i/tau
        c_i = t_i / ((1+b_i) + (1-b_i) * t_i)
        f_i = c_i / (1 - np.sum(c_i, axis=0))
        vector = self.ctrlvector @ f_i
        lon, lat = vectortolatlon(vector).reshape(pts.shape)
        #FIXME maybe transform to/from the authalic latitude here
        return np.stack([lon, lat])#self.authalic(lat)])
    
    def authalic(self, phi, scale = np.pi/180):
        phi = phi * scale
        e=self.geoid.es
        qp = 1 + (1 - e**2)/e * np.arctanh(e)
        q = ((1 - e**2)*np.sin(phi)/(1 - (e*np.sin(phi))**2) + 
             (1 - e**2)/e * np.arctanh(e*np.sin(phi)) )
        return np.arcsin(q/qp)/scale
#%%
