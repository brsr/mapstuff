#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:15:06 2021

@author: brsr
"""

import numpy as np
import matplotlib.pyplot as plt
import mapproj
import fiona
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import geopandas
import pyproj
geod = pyproj.Geod(a=1, f=0)
n = 9
a = np.arctan(1/2)/np.pi*180
actrlpts3 = np.array([[15+0, 15+36, 15-36],
                      [-a, a, a]])
#actrlpts3 = np.array([[ 0, 0, 90],
#                      [90, 0,  0]])
ctrlpoly3 = mapproj.geodesics(actrlpts3[0], actrlpts3[1], geod, includepts=True)
tgtpts3 = mapproj.complex_to_float2d(1j*np.exp(2j/3*np.arange(3)*np.pi)).T
bp = mapproj.Barycentric(tgtpts3)

grid3 = mapproj.Barycentric.grid(1/8)
gridp3 = mapproj.Barycentric.gridpolys(n=9)
#%%
gridbary = mapproj.transeach(bp.transform, gridp3)
conformal = mapproj.ConformalTri3(actrlpts3, tgtpts3)
invframe = mapproj.transeach(conformal.invtransform, gridbary)#slooooow
invframev = mapproj.transeach(mapproj.UnitVector.transform, invframe)
invframe.plot()

#%%
res = geod.inv(actrlpts3[0], actrlpts3[1],
               np.roll(actrlpts3[0], -1), np.roll(actrlpts3[1], -1))
cornerangle = np.pi/180*(res[0] - np.roll(res[1], 1)).mean() #np.pi*2/5 #
edgelength = res[2].mean()

initial = conformal.ctrlpts_v
anglesumtarget = np.ones(shape=(n+1,n+1))
anglesumtarget = np.tril(anglesumtarget, -1)[::-1]
#anglesumtarget[..., 0] = 0
#anglesumtarget[-1] = 0
anglesumtarget[anglesumtarget == 0] = np.nan
ind = np.arange(0,n)
edgeweight = np.ones(n)*2
edgeweight[[0, -1]] = 1
edge1 = (ind, 0)
edge2 = (0, ind)
edge3 = (ind,ind[::-1])
anglesumtarget[edge1] = 1/2
anglesumtarget[edge2] = 1/2
anglesumtarget[edge3] = 1/2
anglesumtarget *= 2*np.pi
anglesumtarget[0, 0] = cornerangle
anglesumtarget[-2, 0] = cornerangle
anglesumtarget[0, -2] = cornerangle

msplitframe = np.array([[0, 1, 2],
                        [2, 0, 1]])
msplit1 = np.tile(msplitframe, (3, n, n))[..., :n,:n]
msplit = (msplit1 + np.arange(3)[:, np.newaxis, np.newaxis]) % 3
msplit = msplit == 0
msplit[:, ~np.isfinite(anglesumtarget[:-1,:-1])] = False
#neighbors like this
#   n n
# n x n
# n n

neighbors = np.array([[ 1,  1,  0, -1, -1,  0],
                      [ 0, -1, -1,  0,  1,  1]])
grindex = np.array(np.meshgrid(ind, ind))

neighborhood = neighbors[..., np.newaxis, np.newaxis] + grindex[:,np.newaxis]

findex = np.array(np.where(np.isfinite(anglesumtarget))).T
r = np.ones(shape=anglesumtarget.shape, dtype=float)*cornerangle/(2*n-2)
r[~np.isfinite(anglesumtarget)] = np.nan
r[[0, -2, 0], [0, 0, -2]] /= 3
#%%
for i in range(128):
    x = r[:-1, :-1]
    y = r[neighborhood[0], neighborhood[1]]
    z = np.roll(y, 1, axis=0)
    if np.any(x+y+z > np.pi):
        break
    locos_x_yz = np.arccos((np.cos(y+z) - np.cos(x+y)*np.cos(x+z))/
                           (np.sin(x+y)*np.sin(x+z)))
    #locos_x_yz = np.arccos(((x+y)**2 + (x+z)**2 - (y+z)**2)/
    #                        (2*(x+y)*(x+z)))
    anglesum = np.nansum(locos_x_yz, axis=0)
    pctdiff = (anglesum/anglesumtarget[:-1,:-1])
    pctdiff /= np.nanmean(pctdiff)
    #pctdiff -= np.clip(pctdiff, 0.9, 1.1)
    #pctdiff /= np.nanmean(pctdiff)
    #ind = np.unravel_index(np.nanargmax(abs(pctdiff)), pctdiff.shape)
    r[:-1, :-1] *= pctdiff
    r *= edgelength/(r[edge1]@edgeweight)
    print(i, np.nanmax(abs(pctdiff-1)))
    if np.nanmax(abs(pctdiff-1)) < 1E-7:
        break
    #print(ind, r[ind], pctdiff[ind])

#print(r[edge1]@edgeweight, edgelength)
print(np.round(r[:-1,:-1], 3))
#%%0.9999999999999746 1.0000000000000149
#%%
for i in range(36*256):
    ind = findex[i % findex.shape[0]]
    x = r[ind[0], ind[1]]
    y = r[neighbors[0] + ind[0], neighbors[1] + ind[1]]
    z = np.roll(y, 1, axis=0)
    locos_x_yz = np.arccos((np.cos(y+z) - np.cos(x+y)*np.cos(x+z))/
                           (np.sin(x+y)*np.sin(x+z)))
    anglesum = np.nansum(locos_x_yz, axis=0)
    pctdiff = anglesum/anglesumtarget[ind[0],ind[1]]#np.clip(, 0.8, 1.2)
    r[ind[0], ind[1]] *= pctdiff
    r *= edgelength/(r[edge1]@edgeweight)
    #print(ind, r[ind[0], ind[1]], pctdiff)

print(r[edge1]@edgeweight, np.pi/2)
print(np.round(r[:-1,:-1], 3))
#%%
vertices = np.ones((3,n+1,n+1))*np.nan
vertices[:,0,0] = initial[:,0]
vertices[:,-2,0] = initial[:,1]
vertices[:,0,-2] = initial[:,2]

r1 = r[edge1]
t = (r1[:-1] + r1[1:]).cumsum()/edgelength
t = np.concatenate([[0,], t])
e1 = mapproj.slerp(initial[:,0], initial[:,1], t[:, np.newaxis]).T
e2 = mapproj.slerp(initial[:,0], initial[:,2], t[:, np.newaxis]).T
e3 = mapproj.slerp(initial[:,2], initial[:,1], t[:, np.newaxis]).T
vertices[:,edge1[0], edge1[1]] = e1
vertices[:,edge2[0], edge2[1]] = e2
vertices[:,edge3[0], edge3[1]] = e3
#%%
for i in range(1, n-1):
    for j in range(1, n-i-1):
        index = np.array([i, j])
        indexnb = index[:,np.newaxis] + neighbors
        vertexnb = vertices[:, indexnb[0], indexnb[1]]
        rnb = r[indexnb[0], indexnb[1]]
        ri = r[i, j]
        filled = np.all(np.isfinite(vertexnb), axis=0)
        vertexnb = vertexnb[:, filled]
        rnb = rnb[filled]
        cl = np.cos(rnb+ri)
        lq = np.linalg.lstsq(vertexnb.T, cl)
        v = lq[0]
        norm = np.linalg.norm(v)
        v /= norm
        vertices[:, i, j] = v
        print(i, j, filled.sum(), lq, norm)

vindex = np.all(np.isfinite(vertices), axis=0)
result = mapproj.UnitVector.invtransform_v(vertices)
#%%
fig, axes = plt.subplots(ncols = 3, figsize=(10, 8), sharex=True, sharey=True)
axes[0].plot(vertices[0], vertices[1])
axes[1].plot(vertices[0], vertices[2])
axes[2].plot(vertices[1], vertices[2])
for ax in axes:
    ax.set_aspect('equal')
#%%
fig, ax = plt.subplots(figsize=(10, 8))
invframe.plot(ax=ax)
ax.scatter(*result, color='k')
ax.scatter(*actrlpts3, color='y')
#%%
triframe = np.array([[[0,0,1],
                 [0,1,0]],
                [[1,0,1],
                 [1,1,0]]])
tris = []
for i in range(n-1):
    for j in range(n-i-1):
        for tf in triframe:
            xy = result[:,i+tf[0], j+tf[1]]
            if np.all(np.isfinite(xy)):
                tris.append(Polygon(xy.T))

gptris = geopandas.GeoSeries(tris)
#use geopandas.intersect to determine which grid cell a point lands in