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

#approximate functions of r as taylor series? as fourier series?
np.seterr(divide='ignore')

geod = pyproj.Geod(ellps='WGS84')
r = (2*geod.a + geod.b)/3
geod = pyproj.Geod(a=r,b=r)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#Hemisphere
#controlpts = geopandas.GeoSeries([Point(-180,0), Point(0, -60), Point(0, 60)])
#Strebe Africa
#controlpts = geopandas.GeoSeries([Point(0,22), Point(22.5, -22), Point(45, 22)])
#Australia:
#controlpts = geopandas.GeoSeries([Point(134, -8), Point(110, -32), Point(158, -32)])
#Canada Atlas:
#controlpts = geopandas.GeoSeries([Point(-98-13/60, 61+39/60), Point(-135, 40), Point(-55, 40)])
#Eastern South America: (basically Brazil)
#controlpts = geopandas.GeoSeries([Point(-63-33/60, 8+8/60),
#                                  Point(-58-33/60, -34-35/60), Point(-35-13/60, -5-47/60)])
#Northwest South America:
#controlpts = geopandas.GeoSeries([Point(-69, -25), Point(-55, 10), Point(-85, 10)])
#Southern South America:
#controlpts = geopandas.GeoSeries([Point(-43, -18), Point(-72, -18), Point(-72, -56) ])
#Europe Wall Map:
#controlpts = geopandas.GeoSeries([Point(15, 72), Point(-8, 33), Point(38, 33)])
#Africa Wall Map:
controlpts = geopandas.GeoSeries([Point(-19-3/60, 24+25/60), Point(20, -35), Point(59+3/60, 24+25/60)])
#Canada Wall Map:
#controlpts = geopandas.GeoSeries([Point(-150, 60), Point(-97.5, 50), Point(-45, 60)])
#North America Wall Map:
#controlpts = geopandas.GeoSeries([Point(-150, 55), Point(-92.5, 10), Point(-35, 55)])
#South America Wall Map:
#controlpts = geopandas.GeoSeries([Point(-80, 9), Point(-71, -53), Point(-35, -6)])
controlpts.crs = world.crs
actrlpts = ptseriestoarray(controlpts)
assert np.linalg.det(latlontovector(actrlpts[0],actrlpts[1])) > 0
antipodes = np.array([actrlpts[0] - 180, -actrlpts[1]])
antipodes[0] = antipodes[0] + np.where(antipodes[0] < -180, 360, 0)
ct = ChambTrimetric(actrlpts, geod)
#ls = LstSqTrimetric(actrlpts, geod)
lt = LinearTrimetric(actrlpts, geod)
print(ct.sidelengths)
print(np.linalg.norm(ct.tgtpts - np.roll(ct.tgtpts, -1, axis=1), axis=0))
print(np.linalg.norm(lt.tgtpts - np.roll(lt.tgtpts, -1, axis=1), axis=0))
tgtpts = lt.tgtpts
#ctgt = tgtpts.T.copy().view(dtype=complex).squeeze()
#rctgt = -ctgt/(ctgt/abs(ctgt))[0]
#tgtpts = rctgt[...,np.newaxis].view(dtype=float).T
#lt.setmat(tgtpts)
#ct.tgtpts = tgtpts
tgtpts_s = tgtpts*1E-3
#have to reverse orientation because Proj's implementation expects clockwise
#false northing to get them to line up right
# projdict = {'proj':'chamb',        #'y_0':-3026660.165922658,
#             'lon_1':actrlpts[0,0], 'lat_1':actrlpts[1,0],
#             'lon_2':actrlpts[0,2], 'lat_2':actrlpts[1,2],
#             'lon_3':actrlpts[0,1], 'lat_3':actrlpts[1,1]}

#world with western hemisphere shoved over so we can plot the antipodes later
def transform(x, y):
    x = np.array(x % 360)
    return x, y
index = (np.isin(world.continent, ["North America", 'Oceania', 'South America'])
         | (world.name == 'Russia'))
win = world[index]
wout = transeach(transform, win.geometry)
wout.index = world.index[index]
world_shift = world.copy()
world_shift.geometry.loc[index] = wout
gd = geodesics(actrlpts[0], actrlpts[1], geod, n=100)
#%%#figure: world map, with points
fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='k')
grat.plot(ax=ax, color='lightgrey', linewidth=1)
controlpts.plot(ax=ax, color='green')
gd.plot(ax=ax, color='green')
#antigd.plot(ax=ax, color='yellow')
ax.axis('equal')

#%%
testpt = [10, 10]
f, b, r = geod.inv(testpt[0]*np.ones(actrlpts.shape[1]),
                   testpt[1]*np.ones(actrlpts.shape[1]),
                   actrlpts[0], actrlpts[1])

az = 0
circs = []
for x, y, ri in zip(actrlpts[0], actrlpts[1], r):
    circ = []
    for az in range(361):
        out = geod.fwd(x, y, az, ri)
        circ.append(out[:2])
    circs.append(LineString(circ))

circs = geopandas.GeoSeries(circs)
#%%#figure: show construction
fig, axes = plt.subplots(1,2, figsize=(10, 5))
ax= axes[0]
#world.plot(ax=ax, color='k')
#grat.plot(ax=ax, color='lightgrey', linewidth=1)
controlpts.plot(ax=ax, color='green')
gd.plot(ax=ax, color='green')#, linestyle=':')
circs.plot(ax=ax, color='b')
#antigd.plot(ax=ax, color='yellow')
ax.set_aspect('equal', 'box')
ax.set_xlim(-25, 75)
ax.set_ylim(-40, 55)
ax = axes[1]
#world_lt.plot(ax=ax, color='k')
#grat2_lt.plot(ax=ax, color='lightgrey', linewidth=1)
#gd_lt.plot(ax=ax, linestyle=':', color='green')
index = [0,1,2,0]
ax.plot(tgtpts_s[0, index], tgtpts_s[1, index], color='green', marker='o')
for xi, yi, ri in zip(tgtpts_s[0], tgtpts_s[1], r/1E3):
    c = plt.Circle((xi, yi), ri, color='b', fill=False)
    ax.add_artist(c)

ri2 = np.roll(r, 1)**2
rj2 = np.roll(r, -1)**2
zi = np.roll(tgtpts, 1, axis=-1)
zj = np.roll(tgtpts, -1, axis=-1)
#should work as long as xi != xj
y = np.array([-5E6, 5E6])[..., np.newaxis]
x = ((ri2 - rj2)/2 + y*(zi[1]-zj[1]))/(zj[0]-zi[0])
ax.plot(x/1E3, y/1E3, color='r')
ax.set_aspect('equal', 'box')
ax.set_xlim(-4.25E3, 5E3)
ax.set_ylim(-4.75E3, 4.5E3)
#ax.set_xlim(-1.4E6,-1.25E6)
#ax.set_ylim(4E5,6E5)
#figure out how to inset this later
fig, ax = plt.subplots( figsize=(10, 1))
#gd_lt.plot(linestyle=':', ax=ax, color='green')
ax.plot(tgtpts_s[0, index], tgtpts_s[1, index], color='green', marker='o')
for xi, yi, ri in zip(tgtpts_s[0], tgtpts_s[1], r/1E3):
    c = plt.Circle((xi, yi), ri, color='b', fill=False)
    ax.add_artist(c)
ctpt = ct.transform(testpt[0], testpt[1])
ltpt = lt.transform(testpt[0], testpt[1])
ax.scatter(ctpt[0]/1E3, ctpt[1]/1E3, color='b')
ax.scatter(ltpt[0]/1E3, ltpt[1]/1E3, color='r')
ax.plot(x/1E3, y/1E3, color='r')
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.3E3,-0.8E3)
ax.set_ylim(100,600)
#%%
def scale(x, y, scale=1E-3):
    return x*scale, y*scale

adegpts = np.array(np.meshgrid(np.linspace(-180,180,361),
                               np.linspace(-90,90,181)))#.transpose(0,2,1)
degpts = arraytoptseries(adegpts)
degpts.crs = world.crs
grat = graticule()
grat2 = graticule(lonrange=[-180,0])#[-100,150])#to avoid the antipodal part
#antigd = geodesics(actrlpts[0] - 180, -actrlpts[1], geod)

controlpts_lt = transeach(lt.transform, controlpts)
world_lt = transeach(lt.transform, world.geometry)
grat_lt = transeach(lt.transform, grat)
grat2_lt = transeach(lt.transform, grat2)
adegpts_lt = lt.transform_v(adegpts)#.reshape((2,-1)))
gd_lt = transeach(lt.transform, gd)
#antigd_lt = transeach(lt.transform, antigd)
adegpts_rt = lt.invtransform_v(adegpts_lt).reshape((adegpts.shape))

controlpts_ct = transeach(ct.transform, controlpts)
world_ct = transeach(ct.transform, world.geometry)
grat_ct = transeach(ct.transform, grat)
grat2_ct = transeach(ct.transform, grat2)
adegpts_ct = ct.transform_v(adegpts)
gd_ct = transeach(ct.transform, gd)
adegpts_rc = lt.invtransform_v(adegpts_ct).reshape((adegpts.shape))


controlpts_lt_s = transeach(scale, controlpts_lt)
world_lt_s = transeach(scale, world_lt)
grat_lt_s = transeach(scale, grat_lt)
grat2_lt_s = transeach(scale, grat2_lt)
adegpts_lt_s = adegpts_lt*1E-3
gd_lt_s = transeach(scale, gd_lt)

controlpts_ct_s = transeach(scale, controlpts_ct)
world_ct_s = transeach(scale, world_ct)
grat_ct_s = transeach(scale, grat_ct)
grat2_ct_s = transeach(scale, grat2_ct)
adegpts_ct_s = adegpts_ct*1E-3
gd_ct_s = transeach(scale, gd_ct)
#%%
omega_lt, scale_lt = omegascale(adegpts, adegpts_lt, actrlpts, tgtpts, geod)
#omega_ls, scale_ls = omegascale(adegpts, degpts_ls, actrlpts, tgtpts, geod)
omega_ct, scale_ct = omegascale(adegpts, adegpts_ct, actrlpts, tgtpts, geod)
#%%#figure: comparison of projections
fig, axes = plt.subplots(1,2, figsize=(10, 5), sharex=True, sharey=True)
ax = axes[0]
#center = (np.nanmax(degpts, axis=(1,2)) + np.nanmin(degpts, axis=(1,2)))/2
world_ct_s.plot(ax=ax, color='k')
grat_ct_s.plot(ax=ax, linewidth=1)
controlpts_ct_s.plot(ax=ax, color='green')
gd_ct_s.plot(ax=ax, color='green')
#antigd_ct.plot(ax=ax, color='yellow')
#ax.scatter(tgtpts[0], tgtpts[1], color='green')
ax = axes[1]
world_lt_s.plot(ax=ax, color='k')
grat_lt_s.plot(ax=ax, linewidth=1)
controlpts_lt_s.plot(ax=ax, color='green')
gd_lt_s.plot(ax=ax, color='green')
#antigd_lt.plot(ax=ax, color='yellow')

#ax.scatter(tgtpts[0], tgtpts[1], color='green')

#%%#figure: zoom comparison of projections
index = [0,1,2,0]
fig, axes = plt.subplots(1,2, figsize=(10, 5), sharex=True, sharey=True)
ax = axes[0]
world_ct_s.plot(ax=ax, color='k')
grat_ct_s.plot(ax=ax, color='lightgrey', linewidth=1)
gd_ct_s.plot(ax=ax, color='green')
controlpts_ct_s.plot(ax=ax, color='green')
#ax.scatter(tgtpts_s[0], tgtpts_s[1], color='green')
#ax.plot(tgtpts_s[0, index], tgtpts_s[1, index], color='green',
#        linestyle=':', marker='o')
#ax.plot(tgtpts_s[0, index], tgtpts_s[1, index], color='green',
#        linestyle=':', marker='o')
ax = axes[1]
world_lt_s.plot(ax=ax, color='k')
grat_lt_s.plot(ax=ax, color='lightgrey', linewidth=1)
gd_lt_s.plot(ax=ax, color='green')
controlpts_lt_s.plot(ax=ax, color='green')
#ax.scatter(tgtpts_s[0], tgtpts_s[1], color='green')
#ax.plot(tgtpts_s[0, index], tgtpts_s[1, index], color='green',
#        linestyle=':', marker='o')
ax.set_xlim(-5E3,5E3)
ax.set_ylim(-5E3,5E3)
#ax.set_ylim(0,1E4)

#%%#figure: transform inv_transform round trip
index = scale_lt > 0
cc = np.concatenate([adegpts, adegpts_rt]).reshape((4,-1)).T
lparam = [geod.line_length([x1,x2],[y1,y2]) for x1, y1, x2, y2 in cc]
param = np.array(lparam).reshape(adegpts.shape[1:])*1E-3
param[~index] = np.nan
x, lev = np.histogram(param, range=(0, 100))
fig, ax = plt.subplots(figsize=(10, 5))
grat.plot(ax=ax, color='lightgrey', linewidth=1)
cs = ax.contour(adegpts[0],adegpts[1], param, levels=lev)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
fig.colorbar(cs)
controlpts.plot(color='green', ax=ax)
#ax.axis('equal')
#%%#figure: transform inv_transform round trip, 
#but forward is chamberlin and reverse is linear
index = scale_ct > 0
cc = np.concatenate([adegpts, adegpts_rc]).reshape((4,-1)).T
lparam = [geod.line_length([x1,x2],[y1,y2]) for x1, y1, x2, y2 in cc]
param = np.array(lparam).reshape(adegpts.shape[1:])*1E-3
#param[~index] = np.nan
x, lev = np.histogram(param, range=(0, 1000))
fig, ax = plt.subplots(figsize=(10, 5))
grat.plot(ax=ax, color='lightgrey', linewidth=1)
cs = ax.contour(adegpts[0],adegpts[1], param, levels=lev)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
fig.colorbar(cs)
controlpts.plot(color='green', ax=ax)
#ax.axis('equal')
#%%#figure: analysis of variable h
ppts = np.array([[0,0],
                 [0, 9E6],
                 [0, 1.8E7],
                 [6E6, 9E6],
                 [6E6, 0],
                 [1.2E7, 0]]).T
hp, nm = lt.nmforplot(ppts)
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
#%%#figure: antipodes
grat3 = graticule(lonrange=[150,255], latrange = [-60, 30])
fig, ax = plt.subplots(figsize=(10, 5), sharex=True, sharey=True)
for scale, linestyle in [(scale_ct, 'dashed'), (scale_lt, 'solid')]:
    #grat.plot(ax)
    cs = ax.contour(adegpts[0], adegpts[1], scale,
                    levels=0, linestyles=linestyle)
ax.scatter(antipodes[0], antipodes[1], color='k', marker='x')
world.plot(ax=ax, color='k')
grat.plot(ax=ax, color='lightgrey', linewidth=1)
#antigd.plot(ax=ax, color='yellow')
#ax.axis('equal')
#ax.set_xlim(155,250)
#ax.set_ylim(-50,30)
#%%#figure: scale
fig, axes = plt.subplots(1,2, figsize=(10, 5), sharex=True, sharey=True)
for scale, degpts, ax in zip([scale_ct, scale_lt],
                         [adegpts_ct_s, adegpts_lt_s], axes):
    index = scale > 0
    param = scale.copy()
    param[~index] = np.nan
    cs = ax.contour(degpts[0], degpts[1], param,
                    levels=[0.9,0.95,1,1.05,1.1,1.15])
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')
    ax.scatter(tgtpts_s[0], tgtpts_s[1], color='k')
    #ax.axis('equal')
ax.set_xlim(-5E3,5E3)
ax.set_ylim(-5E3,5E3)
#ax.set_ylim(0,1E4)

#fig.colorbar(cs)
#%%#figure: omega
fig, axes = plt.subplots(1,2, figsize=(10, 5), sharex=True, sharey=True)
for omega, degpts, ax in zip([omega_ct, omega_lt],
                         [adegpts_ct_s, adegpts_lt_s], axes):
    index = scale > 0
    param = omega.copy()
    param[~index] = np.nan
    cs = ax.contour(degpts[0], degpts[1], param,
                    levels=[1,2,3,5,10,15])
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
    ax.scatter(tgtpts_s[0], tgtpts_s[1], color='k')
    #ax.axis('equal')
ax.set_xlim(-5E3,5E3)
ax.set_ylim(-5E3,5E3)
#ax.set_ylim(0,1E4)

#%%#figure: maximum deviation in distance
lengths = []
for i in range(3):
    f, b, l = geod.inv(adegpts[0], adegpts[1],
                       actrlpts[0,i]*np.ones(adegpts.shape[1:]),
                       actrlpts[1,i]*np.ones(adegpts.shape[1:]))
    lengths.append(l)
lengths = np.array(lengths)
levels = [0,20,50,100,200,500]
fig, axes = plt.subplots(1,2, figsize=(10, 5), sharex=True, sharey=True)
for degpts, ax in zip([adegpts_ct, adegpts_lt], axes):
    x = degpts[:,np.newaxis] - tgtpts[...,np.newaxis,np.newaxis]
    y = sqrt(np.sum(x**2, axis=0))
    result = np.nanmax(abs(y - lengths), axis=0)*1E-3
    cs = ax.contour(adegpts[0],adegpts[1], result,
                    levels=levels)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
    ax.scatter(actrlpts[0], actrlpts[1], color='k')
    #ax.axis('equal')
    #ax.set_xlim(-90, 90)
    #ax.set_ylim(-90, 90)
#%%#figure: distance between two projections (probably omit)
fig, ax = plt.subplots(figsize=(10, 5))
grat.plot(ax=ax, color='lightgrey', linewidth=1)
levels = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
cc = np.concatenate([adegpts_ct, adegpts_lt]).reshape((4,-1)).T
lparam = [sqrt((x1-x2)**2 + (y1-y2)**2) for x1, y1, x2, y2 in cc]
#ylparam = [(y1-y2) for x1, y1, x2, y2 in cc]
param = np.array(lparam).reshape(adegpts.shape[1:])*1E-3
cs = ax.contour(adegpts[0],adegpts[1], param, levels=levels)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')
ax.scatter(actrlpts[0], actrlpts[1], color='k')

#ax.set_xlim(-90, 90)
#ax.set_ylim(-90, 90)
fig.colorbar(cs)
#%%
# m  = np.concatenate([ptseriestoarray(controlpts_ct), np.ones((1,3))])
# m2 = np.concatenate([ptseriestoarray(controlpts_ct2), np.ones((1,3))])
# mx = m2 @ np.linalg.inv(m)
# l0 = geod.line_lengths(list(actrlpts[0]) + [actrlpts[0,0]],
#                        list(actrlpts[1]) + [actrlpts[1,0]])
# l1 = np.linalg.norm(m - np.roll(m,-1,axis=1), axis=0)
# l2 = np.linalg.norm(m2 - np.roll(m2,-1,axis=1), axis=0)
#%%#aesthetic
fig, ax= plt.subplots(figsize=(10, 10), sharex=True, sharey=True)
index = np.isin(world.continent, ['North America', 'South America'])
world_lt_s[index].plot(ax=ax, color='k')
grat2_lt_s.plot(ax=ax, linewidth=1)
ax.axis('off')
