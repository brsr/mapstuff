#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:56:44 2020

@author: brsr
"""

import geopandas
import pandas as pd
#import shapely
#from shapely.geometry import Point, LineString, MultiPolygon, Polygon
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
actrlpts4 = np.array([[0, 90, 90, 0],
                      [-a, -a, a, a]])
ctrlpoly4 = mapproj.geodesics(actrlpts4[0], actrlpts4[1], geod, includepts=True)

antipodepoly3 = mapproj.transeach(mapproj.transform_antipode, ctrlpoly3)
antipodepoly4 = mapproj.transeach(mapproj.transform_antipode, ctrlpoly4)

#cycle3 = [0,1,2,0]
#sidelengths3 = geod.line_lengths(actrlpts3[0][cycle], actrlpts3[1][cycle])
#print(sidelengths3)
#tgtpts3 = mapproj.trigivenlengths(sidelengths3)

adegpts = np.array(np.meshgrid(np.linspace(-180, 180, 361),
                               np.linspace(-90, 90, 181)))
#%%
projs = {#'Conformal': mapproj.ConformalTri3(actrlpts3, tgtpts3),
         #'Linear Trimetric':        mapproj.LinearTrimetric(actrlpts3, geod),
         #'Areal':                   mapproj.Areal(actrlpts3),
         #'Fuller explicit':         mapproj.FullerEq(actrlpts3),
         #'Fuller':                  mapproj.FullerTri(actrlpts3, tweak=False),
         #'Fuller Tweaked':          mapproj.FullerTri(actrlpts3, tweak=True),
         'Bisect':                  mapproj.BisectTri(actrlpts3),#?
         'Bisect2':                  mapproj.BisectTri2(actrlpts3),#?
         #'Snyder Equal-Area 3':     mapproj.SnyderEA3(actrlpts3),
         #'Snyder Symmetrized':      mapproj.SnyderEASym(actrlpts3),#?
         #'Alfredo':                mapproj.Alfredo(actrlpts3),#polygonal?
         #'Alfredo Tweaked':        mapproj.Alfredo(actrlpts3, tweak=True),#not polygonal
         #'SEA':                     mapproj.SnyderEA(actrlpts3),
         #'Crider':                  mapproj.CriderEq(actrlpts4),
         #'Snyder Equal-Area 4':     mapproj.SnyderEA4(actrlpts4),
         }

degpts1 = {}
degpts2 = {}
for name in projs:
    print(name)
    mp = projs[name]
    fwd = mp.transform_v(adegpts)
    inv = mp.invtransform_v(fwd)
    degpts1[name] = fwd
    degpts2[name] = inv
#%%
shape = adegpts.shape[1:]
for name, mp in projs.items():
    d2 = degpts2[name]
    param = np.zeros(shape)
    for i,j in np.ndindex(shape):
        x1, y1 = adegpts[:,i,j]
        x2, y2 = d2[:,i,j]
        p = geod.line_length([x1, x2], [y1, y2])
        param[i,j] = p
    fig, ax = plt.subplots(figsize=(9, 4))
    cs = ax.contourf(*adegpts, param)#, levels=[0,10])    
    fig.suptitle(name)
    fig.colorbar(cs)    
    if mp.nctrlpts == 3:
        ctrlpoly3.plot(ax=ax, color='g')
        antipodepoly3.plot(ax=ax, color='g', linestyle='--')
    elif mp.nctrlpts == 4:
        ctrlpoly4.plot(ax=ax, color='g')
        antipodepoly4.plot(ax=ax, color='g', linestyle='--')
