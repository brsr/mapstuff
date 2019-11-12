#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:06:45 2019

@author: brsr
"""
import geopandas
import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib.pyplot as plt
import numpy as np

geod = pyproj.Geod(ellps='WGS84')
#geod = pyproj.Geod(a=1,b=1)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#controlpts = geopandas.GeoSeries([Point(-180,0), Point(0, -60), Point(0, 60)])
controlpts = geopandas.GeoSeries([Point(0,22), Point(22.5, -22), Point(45, 22)])
#controlpts = geopandas.GeoSeries([Point(0,10), Point(30, 0), Point(40, 0)])
controlpts.crs = {'init': 'epsg:4326'}
actrlpts = np.concatenate([x.xy for x in controlpts], axis=-1)
mp = LinearTrimetric(actrlpts, geod)
#mp = Areal(actrlpts, geod)

adegpts = np.array(np.meshgrid(np.linspace(-180,180,181),
                               np.linspace(-90,90,181)))
degpts = geopandas.GeoSeries([Point(x[0], x[1])
                    for x in adegpts.transpose(1,2,0).reshape(-1, 2)])
degpts.crs = {'init': 'epsg:4326'}
grat = graticule()

world_lintri = transeach(mp.transform, world.geometry)
grat_lintri = transeach(mp.transform, grat)
degpts_lintri = mp.transform_v(adegpts)#.reshape((2,-1)))

world_rt = transeach(mp.invtransform, world_lintri)
grat_rt = transeach(mp.invtransform, grat_lintri)
degpts_rt = mp.invtransform_v(degpts_lintri).reshape((adegpts.shape))

#small polygon parts of russia cross the 180th meridian due to numerical 
#error, put them back on the right side
russia = world_rt.loc[18]
lrussia = list(russia)
for i in (11,12):
    x, y = lrussia[i].exterior.xy
    x = np.array(x)
    x = np.where(x > 0, x - 360, x)
    poly = Polygon(zip(x, y))
    lrussia[i] = poly
russia = MultiPolygon(lrussia)    
world_rt.loc[18] = russia

fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='k')
grat.plot(ax=ax)
ax.axis('equal')

fig, ax = plt.subplots(figsize=(10, 5))
world_lintri.plot(ax=ax, color='k')
grat_lintri.plot(ax=ax)
ax.scatter(mp.tgtpts[0], mp.tgtpts[1], color='green')
ax.axis('equal')

fig, ax = plt.subplots(figsize=(10, 5))
world_rt.plot(ax=ax, color='k')
grat_rt.plot(ax=ax)
controlpts.plot(color='green', ax=ax)
ax.axis('equal')

cc = np.concatenate([adegpts, degpts_rt]).reshape((4,-1)).T
lparam = [geod.line_length([x1,x2],[y1,y2]) for x1, y1, x2, y2 in cc]    
param = np.array(lparam).reshape(adegpts.shape[1:])*1E-3
x, lev = np.histogram(param, range=(0, 100))
fig, ax = plt.subplots(figsize=(10, 5))
cs = ax.contour(adegpts[0],adegpts[1], param, levels=lev)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
fig.colorbar(cs)
controlpts.plot(color='green', ax=ax)
ax.axis('equal')

ppts = np.array([[0,0],
                 [0, 9E6],
                 [0, 1.8E7],
                 [6E6, 9E6],
                 [6E6, 0],
                 [1.2E7, 0]]).T
hp, nm = mp.nmforplot(ppts)
fig, ax = plt.subplots(figsize=(10, 5))
pt = ax.plot(hp.T, nm.T)
ax.axhline(y=1, color='k')
ax.set_ylim([0,2])
ax.set_xlim([0, 8])
#plt.legend(pt)
y = nm.min(axis=1)
x = hp[np.arange(6), nm.argmin(axis=1)]
for i, j, t in zip(x, y, [str(x) for x in ppts.T/1E6]):
    ax.annotate(t, (i, j+0.05), ha='center')
