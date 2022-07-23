#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:33:31 2022

@author: brsr
"""

import geopandas
import pandas as pd
import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mapproj

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
grat = mapproj.graticule()
grat.crs = world.crs
ax = grat.plot(color='k', linewidth=0.5, zorder=0)
world.plot(ax=ax)
#%%
crs = {'proj': 'laea', 'lat_0': 90}
projworld = world[world.continent != 'Antarctica'].to_crs(crs)
projgrat = grat.to_crs(crs)

bounds = projgrat.bounds 
scale = 1/bounds.maxx[bounds.maxx < np.inf].max()
unitworld = projworld.scale(scale, scale, origin=(0,0))
unitgrat = projgrat.scale(scale, scale, origin=(0,0))
#%%
fs = {'identity':   lambda x: 0, 
      'linear_4':   lambda x: -np.pi/4*x, 
      'linear_2':   lambda x: -np.pi/2*x, 
      'linear_1':   lambda x: -np.pi*x,  
      'wiechel_2':  lambda x: -np.arcsin(np.clip(x,-1,1))/2, 
      'wiechel_1':  lambda x: -np.arcsin(np.clip(x,-1,1)), 
      'wiechel_0':  lambda x: -2*np.arcsin(np.clip(x,-1,1)),
      'wiggles_8':  lambda x: np.pi/8*np.sin(np.pi*x),
      'wiggles_16': lambda x: np.pi/16*np.sin(3*np.pi*x),
      'wiggles_32': lambda x: np.pi/32*np.sin(8*np.pi*x),   
      }
for name, f in fs.items():
    def radtransform(x1, y1):
        r = np.sqrt(x1**2 + y1**2)
        theta1 = np.arctan2(y1, x1)
        theta2 = theta1 + f(r)
        x2 = r * np.cos(theta2)
        y2 = r * np.sin(theta2)
        return x2, y2
    tgrat = mapproj.transeach(radtransform, unitgrat)
    tworld = mapproj.transeach(radtransform, unitworld)
    fig, ax = plt.subplots(facecolor='k', figsize=(8, 8))
    #fig.patch.set_alpha(1)
    cc = plt.Circle(( 0 , 0 ), 1 , facecolor='w') 
    ax.add_artist( cc )
    tgrat.plot(ax=ax, color='k', linewidth=0.5, zorder=1)
    tworld.plot(ax=ax, zorder=2)
    ax.set_axis_off()
    fig.savefig(name + '.svg', transparent=True, bbox_inches='tight')
    plt.plot()