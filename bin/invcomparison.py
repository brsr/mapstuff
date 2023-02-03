#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:02:25 2020

@author: brsr
"""
import pyproj
pyproj.datadir.set_data_dir('/usr/local/share/proj')
import fiona
import geopandas
import pandas as pd
#import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar#minimize, root_scalar
import copy

#import os
#os.chdir('Code/mapstuff')
import mapstuff

geod = pyproj.Geod(a=6371, f=0)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
a = np.arctan(1/2)/np.pi*180
actrlpts3 = np.array([[15+0, 15+36, 15-36],
                      [-a, a, a]])
ctrlpoly3 = mapstuff.geodesics(actrlpts3[0], actrlpts3[1], geod, includepts=True)
a = 180/np.pi * np.arctan(1/np.sqrt(2))
actrlpts4 = np.array([[-30, 60, 60, -30],
                      [-a, -a, a, a]])
ctrlpoly4 = mapstuff.geodesics(actrlpts4[0], actrlpts4[1], geod, includepts=True)

ctrlarea3, _ = geod.polygon_area_perimeter(actrlpts3[0],
                                           actrlpts3[1])
ctrlarea4, _ = geod.polygon_area_perimeter(actrlpts4[0],
                                           actrlpts4[1])

tgtpts3 = mapstuff.complex_to_float2d(1j*np.exp(2j/3*np.arange(3)*np.pi)).T
bp = mapstuff.Barycentric(tgtpts3)

grid3 = mapstuff.Barycentric.grid()
grid4 = mapstuff.UV.grid()

gridp3 = mapstuff.Barycentric.gridpolys()
gridp4 = mapstuff.UV.gridpolys()

testshape4 = geopandas.GeoSeries(Polygon(shell=[(0,0),(0.25,0),(0.25,0.25),
                            (0.75,0.25),(0.75,0.5),(0.25,0.5),
                            (0.25,0.75),(1,0.75),(1,1),(0,1)]))
#testshape3 = mapstuff.transeach(bp.invtransform, testshape4)
testshape3 = geopandas.GeoSeries(Polygon(shell=[(1,0,0),
                                                (0.75,0.25,0),
                                                (0.5,0.25,0.25),
                                                (0.5,0.5,0),
                                                (0.25,0.25,0.5),
                                                (0.25,0.75,0),
                                                (0,1,0),
                                                (0,0,1)]))
#%% optimize
projs = {}
nctrlpts = {}
#invbary = {}
invframe = {}
testshapet = {}

projs_k = {'Naive Slerp Tri':  mapstuff.NSlerpTri(actrlpts3, k=1),#add different k vals
           'Naive Slerp Tri~': mapstuff.NSlerpTri(actrlpts3, k=1, exact=False),#add different k vals         
           'Naive Slerp Quad': mapstuff.NSlerpQuad(actrlpts4, k=1),
           'Naive Slerp Quad~': mapstuff.NSlerpQuad(actrlpts4, k=1, exact=False),
           'Naive Slerp Quad 2':  mapstuff.NSlerpQuad2(actrlpts4, k=1),
           'Naive Slerp Quad 2~': mapstuff.NSlerpQuad2(actrlpts4, k=1, exact=False),
           'Elliptical':  mapstuff.EllipticalQuad(actrlpts4, k=1),
           'Elliptical~': mapstuff.EllipticalQuad(actrlpts4, k=1, exact=False),
             }
for name in projs_k:
    mp = projs_k[name]
    i = mp.nctrlpts
    #nctrlpts[name] = i
    if i == 3:
        gridp = gridp3
    else:
        gridp = gridp4
    def objective_a(k):
        mp.k = k
        iv = mapstuff.transeach(mp.invtransform, gridp)
        arealist = []
        for p in iv.geometry:
            area, _ = geod.geometry_area_perimeter(p)
            arealist.append(area)
        return max(arealist)/min(arealist)
    def objective_l(k):
        mp.k = k
        iv = mapstuff.transeach(mp.invtransform, gridp)
        alist = []
        for p in iv.geometry:
            coords = np.array(p.exterior.xy)
            l = geod.line_lengths(coords[0], coords[1])
            aspect = max(l)/min(l)
            alist.append(aspect)            
        return max(alist)
    def objective_l2(k):
        mp.k = k
        iv = mapstuff.transeach(mp.invtransform, gridp)
        alist = []
        for p in iv.geometry:
            coords = np.array(p.exterior.xy)
            l = geod.line_lengths(coords[0], coords[1])
            aspect = max(l)/min(l)
            alist.append(aspect)            
        return np.mean(alist)
    objs = [objective_a, objective_l, objective_l2]
    for obj in objs:
        res = minimize_scalar(obj, bracket=[0,1])
        mp2 = copy.copy(mp)
        mp2.k = res.x
        print(name, res.x)
        if np.round(res.x, 7) not in [0,1]:
            projs[name + ' ' + str(mp2.k)] = mp2
#%%
projs.update({'Areal':               mapstuff.Areal(actrlpts3),
         'Fuller explicit':     mapstuff.FullerEq(actrlpts3),
         #'Fuller':             mapstuff.Fuller(actrlpts3, tweak=False),
         #'Fuller Tweaked':     mapstuff.Fuller(actrlpts3, tweak=True),
         'Bisect':              mapstuff.BisectTri(actrlpts3),
         'Bisect2':             mapstuff.BisectTri2(actrlpts3),
         'Snyder Equal-Area 3': mapstuff.SnyderEA3(actrlpts3),
         #'Snyder Symmetrized': mapstuff.SnyderEASym(actrlpts3),#?
         #'Alfredo':            mapstuff.Alfredo(actrlpts3),#polygonal?
         #'Alfredo Tweaked':    mapstuff.Alfredo(actrlpts3, tweak=True),#not polygonal
         #'SEA':                 mapstuff.SnyderEA(actrlpts3),
         'Reverse Fuller':      mapstuff.ReverseFuller(actrlpts3),
         'Reverse Fuller Tweak': mapstuff.ReverseFuller(actrlpts3, tweak=True),
         'Naive Slerp Tri 0':  mapstuff.NSlerpTri(actrlpts3, k=0),#add different k vals
         'Naive Slerp Tri 1':  mapstuff.NSlerpTri(actrlpts3, k=1),#add different k vals
         'Naive Slerp Tri~ 1': mapstuff.NSlerpTri(actrlpts3, k=1, exact=False),#add different k vals         
         'Crider':              mapstuff.CriderEq(actrlpts4),
         #'Naive Slerp Quad k0': mapstuff.NSlerpQuad(actrlpts4, k=0),
         'Naive Slerp Quad 1': mapstuff.NSlerpQuad(actrlpts4, k=1),
         'Naive Slerp Quad~ 1': mapstuff.NSlerpQuad(actrlpts4, k=1, exact=False),
         'Naive Slerp Quad 2 0':  mapstuff.NSlerpQuad2(actrlpts4, k=0),
         'Naive Slerp Quad 2 1':  mapstuff.NSlerpQuad2(actrlpts4, k=1),
         'Naive Slerp Quad 2~ 1': mapstuff.NSlerpQuad2(actrlpts4, k=1, exact=False),
         'Elliptical 0':  mapstuff.EllipticalQuad(actrlpts4, k=0),
         'Elliptical 1':  mapstuff.EllipticalQuad(actrlpts4, k=1),
         'Elliptical~ 1': mapstuff.EllipticalQuad(actrlpts4, k=1, exact=False),
         'Snyder Equal-Area 4':  mapstuff.SnyderEA4(actrlpts4)
         })

for name in projs:
    print(name)
    mp = projs[name]
    i = mp.nctrlpts
    nctrlpts[name] = i
    #invbary[name] = mapstuff.transeach(mp.invtransform, bary)
    if i == 3:
        invframe[name] = mapstuff.transeach(mp.invtransform, gridp3)
        testshapet[name] = mapstuff.transeach(mp.invtransform, testshape3)
    elif i == 4:
        invframe[name] = mapstuff.transeach(mp.invtransform, gridp4)
        testshapet[name] = mapstuff.transeach(mp.invtransform, testshape4)
#%%
testshapez3 = mapstuff.transeach(bp.transform, testshape3)
gridpz3 = mapstuff.transeach(bp.transform, gridp3)
projs2 = {'Conformal':        mapstuff.ConformalTri3(actrlpts3, tgtpts3),#slow
         #'Linear Trimetric': mapstuff.LinearTrimetric(actrlpts3, geod),#no
         }
         
for name in projs2:
    print(name)
    mp = projs2[name]
    i = mp.nctrlpts
    nctrlpts[name] = i
    #invbary[name] = mapstuff.transeach(mp.invtransform, bary)
    if i == 3:
        invframe[name] = mapstuff.transeach(mp.invtransform, gridpz3)
        testshapet[name] = mapstuff.transeach(mp.invtransform, testshapez3)
    elif i == 4:
        invframe[name] = mapstuff.transeach(mp.invtransform, gridpz4)
        testshapet[name] = mapstuff.transeach(mp.invtransform, testshapez4)
#%%
crs = {'proj': 'longlat', 'datum': 'WGS84'}
crs3= {'proj': 'gnom',
      'lat_0': 10.812316963571709,
      'lon_0': 15}
ctrlpts3 = mapstuff.arraytoptseries(actrlpts3)
ctrlpts3.crs = crs
tgtptsg3 = ctrlpts3.to_crs(crs3)
bg = mapstuff.Barycentric(mapstuff.ptseriestoarray(tgtptsg3))
gridpzz3 = mapstuff.transeach(bg.transform, gridp3)
gridpzz3.crs = crs3
testshapezz3 = mapstuff.transeach(bg.transform, testshape3)
testshapezz3.crs = crs3
name = 'Gnomonic 3'
invframe[name] = gridpzz3.to_crs(crs)
testshapet[name] = testshapezz3.to_crs(crs)
nctrlpts[name] = 3

crs4= {'proj': 'gnom',
      'lat_0': 0,
      'lon_0': 15}
ctrlpts4 = mapstuff.arraytoptseries(actrlpts4)
ctrlpts4.crs = crs
tgtptsg4 = ctrlpts4.to_crs(crs4)
scale = np.array(tgtptsg4[1].xy[0])
def transform_01(x, y, scale=scale):
    return (2*x - 1)*scale, (2*y - 1)*scale
gridpzz4 = mapstuff.transeach(transform_01, gridp4)
gridpzz4.crs = crs4
testshapezz4 = mapstuff.transeach(transform_01, testshape4)
testshapezz4.crs = crs4
name = 'Gnomonic 4'
invframe[name] = gridpzz4.to_crs(crs)
testshapet[name] = testshapezz4.to_crs(crs)
nctrlpts[name] = 4
#%%
ms = ['area', 'lengthrat']#'perim', 'anglediff',
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
    
#%% plots
for name in invframe:
    print(name)
    n = nctrlpts[name]
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
        if n == 3:
            ctrlpoly3.plot(ax=ax, color='g')
        elif n == 4:
            ctrlpoly4.plot(ax=ax, color='g')

        #ax.legend(loc='best')

#%% table
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

ns = np.array([x for x in nctrlpts.values()])
index = ns == 3
#%% efficiency
areas = dat.xs('area', level=1).measure
lens = dat.xs('lengthrat', level=1).measure
areasi = areas[index]
lensi = lens[index]
areasni = areas[~index]
lensni = lens[~index]

efi = np.ones(len(areasi), dtype=bool)
for a,l in zip(areasi, lensi):
    efi[(areasi > a) & (lensi > l)] = False
    
efni = np.ones(len(areasni), dtype=bool)
for a,l in zip(areasni, lensni):
    efni[(areasni > a) & (lensni > l)] = False
    
#%%
for m in ms:
    print(m)
    print(dat.xs(m, level=1)[index][efi].sort_values(['measure', 'max']))
for m in ms:
    print(m)
    print(dat.xs(m, level=1)[~index][efni].sort_values(['measure', 'max']))

#in limit as grid cells get small
#icosahedron:
#grid cells near vertex has interior angles 2pi/5, 3pi/10, 3pi/10
#so by law of sines
#a/sin(2pi/5) = b/sin(3pi/10)
#thus benchmark length ratio is
b3 = np.sin(2*np.pi/5)/np.sin(3*np.pi/10) - 1
cm3 = b3*3/len(gridp3)

#%%
fig, axes = plt.subplots(nrows = 2, figsize=(10, 8))
ax1, ax2 = axes
ax1.scatter(areasi +1, lensi, c=efi)
for n, x, y in zip(areas.index[index][efi], areas[index][efi] + 1, 
                   lens[index][efi]):
    ax1.annotate(n, (x, y), ha='center', va='bottom')
ax2.scatter(areasni +1, lensni, c=efni)
for n, x, y in zip(areas.index[~index][efni], areas[~index][efni] + 1, 
                   lens[~index][efni]):
    ax2.annotate(n, (x, y), ha='center', va='bottom')

ax1.set_xscale('log')
ax2.set_xscale('log')
