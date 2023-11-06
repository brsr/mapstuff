# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:17:21 2023

@author: brsto
"""

import geopandas
import pandas as pd
# import shapely
# from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
#import matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np

# import os
# os.chdir('Code/mapstuff')
import mapstuff

#1
# vabc = np.array([[0, 0, 0.6],
#                   [1, 0, 0],
#                   [0, 1, 0.8]])
#2
# theta = 2, -1.6  # 2*np.pi/3, -np.pi/2
# x = np.cos(theta)
# y = np.sin(theta)
# eps = 1E-2
# vabc = np.array([[1, x[0], x[1]],
#                   [0, y[0], y[1]],
#                   [eps, eps, eps]])
# #3
#x = np.cos(np.pi/36)
#y = np.sin(np.pi/36)
#vabc = np.array([[0, x, x],
#                  [0, -y, y],
#                  [1, 0, 0]])
# #4
vabc = np.random.normal(size=(3,3))
#shared
if np.linalg.det(vabc) < 0:
    vabc = -vabc
vabc /= np.linalg.norm(vabc, axis=0, keepdims=True)
#rotate the whole thing so the triangle is more or less in the middle of the 
#side of the sphere facing us, using the circumcenter
tgt = np.array([0.0, 0.0, 1.0])
vpolar = np.cross(np.roll(vabc, -1, axis=1), np.roll(vabc, 1, axis=1), axis=0)
cc = vpolar.sum(axis=1)
cc /= np.linalg.norm(cc)
costheta = tgt @ cc
axis = np.cross(tgt, cc)
sintheta = np.linalg.norm(axis)
axis /= sintheta
theta = np.arctan2(sintheta,costheta)
K = np.array([[0, -axis[2],axis[1]],
              [axis[2], 0, -axis[0]],
              [-axis[1], axis[0], 0]])
R = np.eye(3) + (np.sin(theta)*K) + ((1-np.cos(theta))*(K@K))
print(np.linalg.det(vabc))
vabc = R.T @ vabc
print(np.linalg.det(vabc))

# %%
area = mapstuff.triangle_solid_angle(vabc[:,0],vabc[:,1],vabc[:,2])

cosabc = np.sum(np.roll(vabc, -1, axis=1) * np.roll(vabc, 1, axis=1), axis=0)
# print(np.arccos(cosabc))
sinabc = np.linalg.norm(vpolar, axis=0, keepdims=True)
# print(np.arcsin(sinabc))
vpolar = np.cross(np.roll(vabc, -1, axis=1), np.roll(vabc, 1, axis=1), axis=0)

vpolar /= sinabc
lonpolar, latpolar = mapstuff.UnitVector.invtransform_v(vpolar)
vv = np.cross(np.roll(vpolar, -1, axis=1), np.roll(vpolar, 1, axis=1), axis=0)
sinpolar = np.linalg.norm(vv, axis=0, keepdims=True)
cospolar = np.sum(np.roll(vpolar, -1, axis=1) * np.roll(vpolar, 1, axis=1),
                  axis=0)
# print(np.arcsin(sinpolar)*180/np.pi)
# print(180*(1 - np.arccos(cospolar)/np.pi))
# print((np.roll(f,1) - np.roll(b,-1)) % 360)
angles = np.pi - np.arccos(cospolar)
degangles = angles*180/np.pi
print(degangles, area*180/np.pi)
# vv /= sinpolar #== vabc
vo = vabc.sum(axis=1)
vo /= np.linalg.norm(vo)
# %% verify center
h = 1/np.sin(angles - area/3)
vc = (vabc*sinabc) @ h
vc /= np.linalg.norm(vc)
#sanity checks
print('h:', h)
print('h*sinabc:', sinabc.squeeze()*h)
print('expect', area/3)
for i in range(3):
    print(mapstuff.triangle_solid_angle(vabc[:, i-1], vabc[:, i], vc))
# %% plot
fig = plt.figure()
ax = plt.axes()
patch = mpl.patches.Circle((0,0), radius=1, fill = False, edgecolor='k')
ax.add_artist(patch)
ax.set_xlim(-1.05,1.05)
ax.set_ylim(-1.05,1.05)
ax.set_aspect('equal')
ax.axis('off')

t = np.linspace(0,1)[:,np.newaxis,np.newaxis]
#this is easier than figuring out how to make broadcasting cooperate
vabcedges = np.zeros((len(t),3,3))
for i in range(3):
    vabcedges[...,i] = mapstuff.slerp(vabc[:,i-1], vabc[:,(i+1)%3], t).squeeze()
#again, forget broadcasting
vms = np.zeros((len(t),3,3))
for i in range(3):
    vms[..., i] = mapstuff.slerp(vc, vabc[:,i], t).squeeze()

ax.scatter(vabc[0], vabc[1], color='b')
ax.plot(vabcedges[:,0], vabcedges[:,1], color='b')
ax.plot(vms[:,0], vms[:,1], color='r')

ax.scatter(vc[0], vc[1], color='r', zorder=5)# label='centroid',
ax.set_title('vertex angles {0:.0f}°, {1:.0f}°, {2:.0f}°'.format(*degangles))
fig.savefig('area_center_{0:.0f}-{1:.0f}-{2:.0f}.svg'.format(*degangles), bbox_inches = 'tight')
