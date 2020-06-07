#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:01:41 2020

@author: brsr
"""

import geopandas
import pandas as pd
#import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#import os
#os.chdir('Code/mapproj')
import mapproj

geod = pyproj.Geod(a=6371, f=0)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
a = np.arctan(1/2)/np.pi*180
actrlpts3 = np.array([[15+0, 15+36, 15-36],
                      [-a, a, a]])
ctrlpoly3 = mapproj.geodesics(actrlpts3[0], actrlpts3[1], geod, includepts=True)

ctrlarea3, _ = geod.polygon_area_perimeter(actrlpts3[0],
                                           actrlpts3[1])

tgtpts3 = mapproj.complex_to_float2d(-1j*np.exp(2j/3*np.arange(3)*np.pi)).T
bp = mapproj.Barycentric(tgtpts3)
vctrlpts3 = mapproj.UnitVector.transform_v(actrlpts3)
#%%
lon, lat = [0, 0]
vtestpt = mapproj.UnitVector.transform(lon, lat)
aa = []
vctrlpts = vctrlpts3
actrlpts = actrlpts3
midpoint_v = np.roll(vctrlpts, 1, axis=1) + np.roll(vctrlpts, -1, axis=1)
midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
midpoint = mapproj.UnitVector.invtransform_v(midpoint_v)
for i in range(3):
    vc = np.roll(vctrlpts, i, axis=1)
    ac = np.roll(actrlpts, i, axis=1)
    mi = midpoint[:,-i]
    lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                      np.cross(vc[..., 0], vtestpt))
    lllproj = mapproj.UnitVector.invtransform_v(lproj)
    dist1x = mapproj.central_angle(vc[..., 1], lproj)
    f, b, dist1x = geod.inv(mi[0], mi[1],
                                 lllproj[0],lllproj[1])
    f0, b0, distm2 = geod.inv(mi[0], mi[1],
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
#return self._fix_corners(lon, lat, betax)
#%%
beta = np.array([1,2,3])/6
#beta = np.ones(3)/3
#beta = np.array([0,1,2])/3
#beta1, beta2, beta3 = testb
#x = beta1/(1 - beta3)
# a = x * ctrlarea3
# pt0 = vctrlpts3[:,0]
# pt1 = vctrlpts3[:,1]
# pt2 = vctrlpts3[:,2]
# cosw = pt1 @ pt2
# w = np.arccos(cosw)
# sinw = np.sin(w)
# p2 = ((np.cos(a/2)* pt2 @ np.cross(pt0, pt1)- np.sin(a/2)*pt2 @ (pt1 + pt0))
#  + np.sin(a/2)*cosw*(1 + pt1 @ pt0))
# p3 = sinw*np.sin(a/2)*(1 + pt0 @ pt1)
# r = 2*p3*p2/(p2**2 - p3**2)
# ts = np.arctan(r)/w

#vctrlpts3 = self.ctrlpts_v
xs = []
ptts = []
for i in range(3):
    beta1, beta2, beta3 = np.roll(beta, -i, axis=0)
    x = beta2/(1 - beta1)
    xs.append(x)
    a = x * ctrlarea3/geod.a**2
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
    print(x, t)
    t = x
    ptt = mapproj.slerp(pt2, pt1, t)
    ptts.append(ptt)

ptts = np.array(ptts).T
pttsll = mapproj.UnitVector.invtransform_v(ptts)

xs = np.array(xs)
tp = np.roll(tgtpts3, 1, axis=1)*(1-xs) + np.roll(tgtpts3,-1, axis=1)*xs

ns = np.cross(vctrlpts3, ptts, axis=0)
nsll = mapproj.UnitVector.invtransform_v(ns)
pts = np.cross(ns, np.roll(ns, -1, axis=1), axis=0)
ptsll = mapproj.UnitVector.invtransform_v(pts)
pt = pts.sum(axis=1)
ptll = mapproj.UnitVector.invtransform_v(pt)

bis = []
for i in range(3):
    bi = geod.npts(pttsll[0,i], pttsll[1,i],
                    actrlpts3[0,i], actrlpts3[1,i], 100)
    bis.append(bi)
bis = np.array(bis)
#%%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
ax1.plot(*tgtpts3[:,[0,1,2,0]])
ax1.scatter(*tp)
ax1.scatter(*bp.transform_v(beta))
ax1.plot([tp[0],tgtpts3[0]],[tp[1],tgtpts3[1]])
ctrlpoly3.plot(ax=ax2)
ax2.scatter(*pttsll)
#ax2.scatter(*nsll)
ax2.scatter(*ptll)
ax2.scatter(*ptsll)
ax2.plot(*bis.T)
