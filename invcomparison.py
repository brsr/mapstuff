#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:02:25 2020

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
a = 180/np.pi * np.arctan(1/np.sqrt(2))
actrlpts4 = np.array([[-30, 60, 60, -30],
                      [-a, -a, a, a]])
ctrlpoly4 = mapproj.geodesics(actrlpts4[0], actrlpts4[1], geod, includepts=True)

ctrlarea3, _ = geod.polygon_area_perimeter(actrlpts3[0],
                                           actrlpts3[1])
ctrlarea4, _ = geod.polygon_area_perimeter(actrlpts4[0],
                                           actrlpts4[1])

tgtpts3 = mapproj.complex_to_float2d(1j*np.exp(2j/3*np.arange(3)*np.pi)).T
bp = mapproj.Barycentric(tgtpts3)

grid3 = mapproj.Barycentric.grid()
grid4 = mapproj.UV.grid()

gridp3 = mapproj.Barycentric.gridpolys()
gridp4 = mapproj.UV.gridpolys()

testshape4 = geopandas.GeoSeries(Polygon(shell=[(0,0),(0.25,0),(0.25,0.25),
                            (0.75,0.25),(0.75,0.5),(0.25,0.5),
                            (0.25,0.75),(1,0.75),(1,1),(0,1)]))
#testshape3 = mapproj.transeach(bp.invtransform, testshape4)
testshape3 = geopandas.GeoSeries(Polygon(shell=[(1,0,0),
                                                (0.75,0.25,0),
                                                (0.5,0.25,0.25),
                                                (0.5,0.5,0),
                                                (0.25,0.25,0.5),
                                                (0.25,0.75,0),
                                                (0,1,0),
                                                (0,0,1)]))

#%%
projs = {#'Conformal':          mapproj.ConformalTri3(actrlpts3, tgtpts),#slow
         #'Linear Trimetric':   mapproj.LinearTrimetric(actrlpts3, geod),#no
         'Areal':               mapproj.Areal(actrlpts3),
         'Fuller explicit':     mapproj.FullerEq(actrlpts3),
         #'Fuller':             mapproj.Fuller(actrlpts3, tweak=False),
         #'Fuller Tweaked':     mapproj.Fuller(actrlpts3, tweak=True),
         'Bisect':              mapproj.BisectTri(actrlpts3),
         'Bisect2':             mapproj.BisectTri2(actrlpts3),
         'Snyder Equal-Area 3': mapproj.SnyderEA3(actrlpts3),
         #'Snyder Symmetrized': mapproj.SnyderEASym(actrlpts3),#?
         #'Alfredo':            mapproj.Alfredo(actrlpts3),#polygonal?
         #'Alfredo Tweaked':    mapproj.Alfredo(actrlpts3, tweak=True),#not polygonal
         #'SEA':                 mapproj.SnyderEA(actrlpts3),
         'Reverse Fuller':     mapproj.ReverseFuller(actrlpts3),
         'Naive Slerp Tri':     mapproj.NSlerpTri(actrlpts3),#add different k vals
         'Crider':              mapproj.CriderEq(actrlpts4),
         'Naive Slerp Quad':    mapproj.NSlerpQuad(actrlpts4),
         'Naive Slerp Quad 2':  mapproj.NSlerpQuad2(actrlpts4),
         'Elliptical':          mapproj.EllipticalQuad(actrlpts4),
         'Snyder Equal-Area 4':  mapproj.SnyderEA4(actrlpts4)
         }

nctrlpts = {}
#invbary = {}
invframe = {}
testshapet = {}
for name in projs:
    print(name)
    mp = projs[name]
    i = mp.nctrlpts
    nctrlpts[name] = i
    #invbary[name] = mapproj.transeach(mp.invtransform, bary)
    if i == 3:
        invframe[name] = mapproj.transeach(mp.invtransform, gridp3)
        testshapet[name] = mapproj.transeach(mp.invtransform, testshape3)
    elif i == 4:
        invframe[name] = mapproj.transeach(mp.invtransform, gridp4)
        testshapet[name] = mapproj.transeach(mp.invtransform, testshape4)

#%%
#atotal, ptotal = geod.polygon_area_perimeter(*actrlpts)
areas = {}
perims = {}
angles = {}
lengths = {}
cycle3 = [0, 1, 2, 0]
cycle4 = [0, 1, 2, 3, 0]
for name in invframe:
    iv = invframe[name]
    arealist = []
    perimlist = []
    anglelist = []
    lengthlist = []
    i = nctrlpts[name]
    for p in iv.geometry:
        area, perim = geod.geometry_area_perimeter(p)
        arealist.append(area)
        perimlist.append(perim)
        coords = np.array(p.exterior.xy)#[:]
#        cycle = cycle3 if i == 3 else cycle4
        l = geod.line_lengths(coords[0], coords[1])
        f, b, _ = geod.inv(coords[0], coords[1],
                           np.roll(coords[0], -1), np.roll(coords[1], -1))
        angle = (np.roll(f, 1) - np.roll(b, -1)) % 360
        anglelist.append(angle)
        lengthlist.append(l)
    ctrlarea =  ctrlarea3 if i == 3 else ctrlarea4
    areas[name] = np.array(arealist)/ctrlarea*len(iv) - 1
    perims[name] = np.array(perimlist)
    angles[name] = np.array(anglelist)
    lengths[name] = np.array(lengthlist)

anglediff = {}
lengthrat = {}
for name in lengths:
    angle = angles[name]
    anglediff[name] = angle.max(axis=1)
    length = lengths[name]
    lengthrat[name] = length.max(axis=1)/length.min(axis=1) - 1

#ms = ['Areas', 'Perimeters', 'Angles', 'Lengths']
for name in invframe:
    iv = invframe[name]
    iv = geopandas.GeoDataFrame(geometry=iv.geometry, data={
                                'area': areas[name],
                                'perim': perims[name],
                                'anglediff': anglediff[name],
                                'lengthrat': lengthrat[name]})
    invframe[name] = iv
#%%
ms = ['area', 'lengthrat']#'perim', 'anglediff',
for name, mp in projs.items():
    ts = testshapet[name]
    ib = invframe[name]

    fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
    fig.suptitle(name)
    ax = axes[0]

    ts.plot(ax=ax)
    #ib.plot(ax=ax, facecolor=None, edgecolor='k')

    axes1 = axes[1:]
    for mn, ax in zip(ms, axes1):
        ib.plot(column=mn, ax=ax, legend=True)
        ax.set_title(mn)

    for ax in axes:
        if mp.nctrlpts == 3:
            ctrlpoly3.plot(ax=ax, color='g')
        elif mp.nctrlpts == 4:
            ctrlpoly4.plot(ax=ax, color='g')

        #ax.legend(loc='best')

#%%
projnames = areas.keys()
index = pd.MultiIndex.from_product([projnames, ms],
                                   names=['Projection', 'Measure'])
cols = ['min', 'max', 'measure']#'q1', 'q99',
dat = pd.DataFrame(index = index, columns=cols)

for name, iv in invframe.items():
    a = iv['area']
    dat.loc[name, 'area'] = [a.min(), a.max(),
                             (a.max() + 1) / (a.min() + 1) - 1]
    b = iv.lengthrat
    dat.loc[name, 'lengthrat'] = [b.min(), b.max(), b.mean()]

for m in ms:
    print(m)
    print(dat.xs(m, level=1).sort_values('measure'))

#in limit as grid cells get small
#grid cells near vertex has interior angles 2pi/5, 3pi/10, 3pi/10
#so by law of sines
#a/sin(2pi/5) = b/sin(3pi/10)
#thus benchmark length ratio is
b = np.sin(2*np.pi/5)/np.sin(3*np.pi/10) - 1
