#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:54:43 2020

@author: brsr
"""
import geopandas
#import pandas as pd
#import shapely
#from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

#import os
#os.chdir('Code/mapstuff')
import mapstuff
cycle = [0,1,2,0]

geod = pyproj.Geod(a=6371, f=0)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

grat = mapstuff.graticule()

#adegpts = np.array(np.meshgrid(np.linspace(-180, 180, 36, endpoint=False),
#                               np.linspace(-90, 90, 18, endpoint=False)))+5
adegpts = np.array(np.meshgrid(np.linspace(-40, 30, 8),
                               np.linspace(-90, 90, 18, endpoint=False)))+5

icosa = mapstuff.multipart.icosa
coords = icosa.sph_tri.transpose(1,0,2)

fig, ax = plt.subplots(figsize=(10, 5))
#world.plot(ax=ax, color='k')
#grat.plot(ax=ax, color='lightgrey')
ax.plot(adegpts[0],adegpts[1])#, c=n)
ax.plot(coords[0,:,cycle],coords[1,:,cycle], c='r')    
ax.axis('equal')
fig.suptitle('Plate Carree')
#%%
worlds = {}
grats = {}
ctrlpts = {}
degptss = {}

pSymSmallCircle = partial(mapstuff.SnyderEASym, p=mapstuff.SmallCircleEA)
icoprojs = {'Areal': mapstuff.IcosahedralProjection(mapstuff.Areal),
            'Trilinear': mapstuff.IcosahedralProjection(mapstuff.Trilinear),
            #'Bisect': mapstuff.IcosahedralProjection(mapstuff.BisectTri),
            #'Bisect 2': mapstuff.IcosahedralProjection(mapstuff.BisectTri2),            
            #'Split Area': mapstuff.IcosahedralProjection(mapstuff.SplitAreaTri),
            #'Split Length': mapstuff.IcosahedralProjection(mapstuff.SplitLengthTri),#?
            #'Split Angle': mapstuff.IcosahedralProjection(mapstuff.SplitAngleTri),
            # 'Snyder Sym': mapstuff.IcosahedralProjection(mapstuff.SnyderEASym),
            # 'Small Circle Sym': mapstuff.IcosahedralProjection(pSymSmallCircle),
            # 'Gnomonic': mapstuff.IcosahedralProjection(mapstuff.TriGnomonic),
            # 'Fuller': mapstuff.IcosahedralProjection(mapstuff.FullerEq),
            # 'Conformal': mapstuff.IcosahedralProjectionNoPost(mapstuff.ConformalTri),
            # 'Snyder C': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA),
            # 'Snyder M': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA, first=1),
            # 'Snyder V': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA, first=2),
            # 'Small Circle C': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA),
            # 'Small Circle M': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA, first=1),
            # 'Small Circle V': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA, first=2),
             }
# icoprojs['O Flation'] = mapstuff.Double(
#     icoprojs['Small Circle Sym'],
#     icoprojs['Snyder Sym'], t = 1.95)#2)
# icoprojs['O Angle'] = mapstuff.Double(
#     icoprojs['Areal'],
#     icoprojs['Fuller'], t = -5.98)
# icoprojs['O Cusps'] = mapstuff.Double(
#     icoprojs['Fuller'],
#     icoprojs['Snyder Sym'], t = 1.41)
# plist = [icoprojs[x] for x in ['Snyder Sym',
#                                 'Small Circle Sym',
#                                 'Fuller',
#                                 'Bisect 2',
#                                 'Areal',
#                                 'Bisect',
#                                 'Gnomonic'
#                                 ]]
# opt_f = np.array([ 2.192, -1.062, -1.14 ,  0.376,  1.508, -0.784, -0.09])
# opt_a = np.array([-0.219, 9.838, -0.77 , -5.603, -13.668,  7.637, 3.785])
# opt_c = np.array([ 1.416, -0.201, -0.371, -0.106, -0.008,  0.203, 0.067])
# icoprojs['MO Flation'] = mapstuff.Multiple(plist, opt_f)
# icoprojs['MO Angle'] = mapstuff.Multiple(plist, opt_a)
# icoprojs['MO Cusps'] = mapstuff.Multiple(plist, opt_c)

for name in icoprojs:
    print(name)
    mp = icoprojs[name]
    worlds[name] = mapstuff.transeach(mp.transform, world.geometry)
    grats[name] = mapstuff.transeach(mp.transform, grat)
    ctrlpts[name] = mp.transform_v(coords)
    degptss[name] = mp.transform_v(adegpts)
#%%
tgt = icosa.tgt_tri.transpose(1,0,2)
#n = np.arange(adegpts.shape[1]*adegpts.shape[2]).reshape(adegpts.shape[1:])
for name in degptss:
    print(name)
    #mp = projs[name]
    degpts = degptss[name]
    ctrlpts_t = ctrlpts[name]
    diff = tgt-ctrlpts_t
    dx = np.sqrt(diff[0]**2 + diff[1]**2)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle(name)
    ax.set_xlim(-1800, 44000)
    ax.set_ylim(-1000, 21000)
    ax.plot(degpts[0],degpts[1], marker='o')#, c=n)
    ax.plot(ctrlpts_t[0,:,cycle],ctrlpts_t[1,:,cycle], c='r')    
    
#%%
for name in worlds:
    print(name)
    #mp = projs[name]
    world_t = worlds[name]
    grat_t = grats[name]
    ctrlpts_t = ctrlpts[name]
    diff = tgt-ctrlpts_t
    dx = np.sqrt(diff[0]**2 + diff[1]**2)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle(name)
    ax.set_xlim(-1800, 44000)
    ax.set_ylim(-1000, 21000)
    grat_t.plot(ax=ax, color='lightgrey')
    world_t.plot(ax=ax, color='k')
    #ax.plot(tgt[0,:,cycle],tgt[1,:,cycle], c='r')
    cs = ax.scatter(tgt[0],tgt[1], c=dx)    
    fig.colorbar(cs)
    fig.savefig('docs/' + name + '.svg')
