#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:18:02 2019

@author: brsr
"""

import geopandas
import pandas as pd
import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib.pyplot as plt
import numpy as np

geod = pyproj.Geod(a=1, b=1)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#controlpts = geopandas.GeoSeries([Point(-179,0), Point(0, -60), Point(0, 60)])
#controlpts = geopandas.GeoSeries([Point(0,22), Point(22.5, -22), Point(45, 22)])
#controlpts = geopandas.GeoSeries([Point(0,10), Point(30, 0), Point(40, 0)])
#controlpts = geopandas.GeoSeries([Point(0, 90), Point(0,0), Point(90, 0)])
#controlpts = geopandas.GeoSeries([Point(-25,40), Point(17.5, -40), Point(60, 40)])
#controlpts = geopandas.GeoSeries([Point(15.5, 90), Point(-29.5,0), Point(60.5, 0)])
#controlpts = geopandas.GeoSeries([Point(-30, 70), Point(-20,0), Point(50, 0)])
#controlpts = geopandas.GeoSeries([Point(0, 90), Point(-25,20),
#                                  Point(50.23502428132136, 20)])
#controlpts = geopandas.GeoSeries([Point(0, 90), Point(-30,0), Point(60, 0)])

#actrlpts = ptseriestoarray(controlpts)
actrlpts = np.array([[ 0, -30, 60],
                     [90,   0,  0]])
controlpts = arraytoptseries(actrlpts)
controlpts.crs = {'init': 'epsg:4326'}

mp = FullerTri(actrlpts)
#ar = Areal(actrlpts)
tgtpts = trigivenlengths(mp.sides)
ct = tgtpts.T.view(dtype=complex).squeeze()
ct = ct *1j*np.exp(-1j*np.angle(ct[0]))
tgtpts = ct[:,np.newaxis].view(dtype=float).T
bp = Barycentric(tgtpts)

adegpts = np.array(np.meshgrid(np.linspace(-180,180,361),
                               np.linspace(-90,90,181)))
#adegpts = np.array(np.meshgrid(np.linspace(-180,0,181),
#                               np.linspace(-90,89,181)))
#adegpts = np.array(np.meshgrid(np.linspace(-30,60,91),
#                               np.linspace(0,90,91)))

degpts = arraytoptseries(adegpts)
degpts.crs = {'init': 'epsg:4326'}
grat = graticule()
grat2 = graticule(spacing2=0.1,
                 lonrange = [-30, 60], latrange = [0, 90])

ctrlboundary = geodesics(actrlpts[0], actrlpts[1], geod)

ctrlpoly = geopandas.GeoSeries(pd.concat([ctrlboundary, controlpts], 
                                            ignore_index=True), 
                                  crs=ctrlboundary.crs)
#%%
fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='k')
grat.plot(ax=ax)
grat2.plot(ax=ax, color='g')
ctrlpoly.plot(ax=ax, color='g')
ax.axis('equal')
#%% forward transform
world_lintrib = transeach(mp.transform, world.geometry)
grat_lintrib = transeach(mp.transform, grat)
grat2_lintrib = transeach(mp.transform, grat2)
ctrlpoly_lintrib = transeach(mp.transform, ctrlpoly)
degpts_lintrib = mp.transform_v(adegpts)

world_lintri = transeach(bp.transform, world_lintrib)
grat_lintri = transeach(bp.transform, grat_lintrib)
grat2_lintri = transeach(bp.transform, grat2_lintrib)
ctrlpoly_lintri = transeach(bp.transform, ctrlpoly_lintrib)
degpts_lintri = bp.transform_v(degpts_lintrib)
#%% omega, scale
omega_lt, scale_lt = omegascale(adegpts, degpts_lintri, actrlpts, tgtpts, geod)
#%% round trip
world_rt = transeach(mp.invtransform, world_lintrib)
grat_rt = transeach(mp.invtransform, grat_lintrib)
degpts_rt = mp.invtransform_v(degpts_lintrib).reshape((adegpts.shape))

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

#%%
#areas = np.array([x.area for x in world_lintri])
#index = areas < 1E20
fig, ax = plt.subplots(figsize=(10, 10))
world_lintri.plot(ax=ax, color='k')
grat2_lintri.plot(ax=ax, color='g')
#ax.scatter(tgtpts[0], tgtpts[1], color='g')
ctrlpoly_lintri.plot(ax=ax, color='g')
#ax.set_xlim(-1E7,1E7)
#ax.set_ylim(-5E6,5E6)
#ax.axis('equal')
#%%
fig, ax = plt.subplots(figsize=(10, 5))
world_rt.plot(ax=ax, color='k')
grat_rt.plot(ax=ax)
ctrlpoly.plot(color='green', ax=ax)
ax.axis('equal')
#%%
cc = np.concatenate([adegpts, degpts_rt]).reshape((4,-1)).T
lparam = [geod.line_length([x1,x2],[y1, y2]) for x1, y1, x2, y2 in cc]
param = np.array(lparam).reshape(adegpts.shape[1:])*6371#*1E-3
index = scale_lt < 0
#param[index] = np.nan
x, lev = np.histogram(param, range=(0, 1))
fig, ax = plt.subplots(figsize=(10, 5))
cs = ax.contour(adegpts[0],adegpts[1], param)#, levels=[1,100,1000])#, levels=lev)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
fig.colorbar(cs)
#controlpts.plot(color='green', ax=ax)
ctrlpoly.plot(ax=ax, color='g')
#ax.plot(actrlpts[0]-180, -actrlpts[1], color='k')
#ax.set_xlim(0,45)
#ax.set_ylim(-22,22)

#ax.axis('equal')
#print(adegpts[:,param > 1])
#%%
ctrl_mp = np.array([mp.transform(*x) for x in actrlpts.T]).T
ctrl_b = np.array([bp.transform(*x) for x in ctrl_mp.T]).T
#%%#figure: scale
fig, ax = plt.subplots(figsize=(10, 5))
index = scale_lt >= 0
param = scale_lt.copy()#*1E12
#param[~index] = np.nan
#ax.scatter(actrlpts[0], actrlpts[1], color='k')
#ax.scatter([140, 130], [0, 70], color='k')

cs = ax.contourf(adegpts[0], adegpts[1], param,
                levels=np.linspace(0,3,11))#[0.9,0.95,1,1.05,1.1,1.15])
#ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')
#ax.axis('equal')
#ax.set_xlim(-5E3,5E3)
#ax.set_ylim(-5E3,5E3)
#ax.set_ylim(0,1E4)
ctrlpoly.plot(ax=ax, color='g')
fig.colorbar(cs)

#%%#figure: omega
fig, ax = plt.subplots(figsize=(10, 5))
#index = scale > 0
param = omega_lt.copy()
#param[~index] = np.nan
cs = ax.contourf(adegpts[0], adegpts[1], param,
                levels=[1,2,3,5,10,15,30,60,90])
#ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
#ax.scatter(actrlpts[0], actrlpts[1], color='k')
ctrlpoly.plot(ax=ax, color='g')
#ax.axis('equal')
#ax.set_xlim(-5E3,5E3)
#ax.set_ylim(-5E3,5E3)
#ax.set_ylim(0,1E4)
fig.colorbar(cs)
#%%#figure: maximum deviation in distance
lengths = []
index = scale_lt >= 0
for i in range(3):
    f, b, l = geod.inv(adegpts[0], adegpts[1],
                       actrlpts[0,i]*np.ones(adegpts.shape[1:]),
                       actrlpts[1,i]*np.ones(adegpts.shape[1:]))
    lengths.append(l)
lengths = np.array(lengths)
levels = np.linspace(0.7,1.5,9)
fig, ax = plt.subplots(figsize=(10, 5))
x = degpts_lintri[:,np.newaxis] - tgtpts[...,np.newaxis,np.newaxis]#?
y = sqrt(np.sum(x**2, axis=0))
y[np.isnan(y)] = 1
result = np.nanmax(y/lengths, axis=0)
index = scale_lt > 0
result[~index] = np.nan
cs = ax.contour(adegpts[0],adegpts[1], result, levels=levels)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.1f')
#ctrlpoly.plot(ax=ax, color='g')
#ax.axis('equal')
#ax.set_xlim(-90, 90)
#ax.set_ylim(-90, 90)
fig.colorbar(cs)
#%% k
bx = np.nditer([degpts_lintrib[0], degpts_lintrib[1], degpts_lintrib[2]])

k = np.array([mp._k_eq(*b)[2] for b in bx]).T.reshape(degpts_lintrib.shape[1:])
index = scale_lt < 0
#param[index] = np.nan
k2 = k.copy()
k2[index] = np.nan

fig, ax = plt.subplots(figsize=(10, 5))
cs = ax.contourf(adegpts[0], adegpts[1], k2)
ax.scatter(actrlpts[0], actrlpts[1], color='k')
fig.colorbar(cs)

#fig.colorbar(cs)
