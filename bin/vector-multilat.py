#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mapproj import rodrigues

northpole = np.array([0,0,1])
alltheta = np.linspace(0, 2*np.pi, 360)
def plotinit():
    fig = plt.figure()
    ax = plt.axes()
    patch = mpl.patches.Circle((0,0), radius=1, fill = False, edgecolor='k')
    ax.add_artist(patch)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax
#%% 2
pt1 = np.array([-0.5,0.25,np.sqrt(0.6875)])

someplane = np.cross(pt1, northpole)
someplane /= np.linalg.norm(someplane)
startpt1 = rodrigues(someplane, pt1, 30*np.pi/180)

circle1 = rodrigues(pt1[np.newaxis], startpt1,
                    alltheta[:, np.newaxis])
pt2 = -pt1
pt2[2] = pt1[2]
startpt2 = rodrigues(someplane, pt2, 37.98*np.pi/180)#45*np.pi/180)#10*np.pi/180)#

circle2 = rodrigues(pt2[np.newaxis], startpt2,
                    alltheta[:, np.newaxis])

cxc = np.cross(pt1, pt2)
ncxc = np.linalg.norm(cxc)
cdotc = pt1 @ pt2
cosr1 = pt1 @ startpt1
cosr2 = pt2 @ startpt2
h1 = (cosr1 - cosr2 * cdotc)/(cxc @ cxc)
h2 = (cosr2 - cosr1 * cdotc)/(cxc @ cxc)
t = np.sqrt(1 - h1**2 - h2**2 - 2*h1*h2*cdotc)/ncxc
#if np.isnan(t):
#    t = 0
vax = h1 * pt1 + h2 * pt2
v1 = vax + t * cxc
v2 = vax - t * cxc
v = np.stack([v1, v2])
v /= np.linalg.norm(v, axis=-1, keepdims=True)
vax /= np.linalg.norm(vax, axis=-1, keepdims=True)
fig, ax = plotinit()
ax.scatter(pt1[0], pt1[1], color='r')
ax.plot(circle1[..., 0], circle1[..., 1], color='r')
ax.scatter(pt2[0], pt2[1], color='b')
ax.plot(circle2[..., 0], circle2[..., 1], color='b')
ax.scatter(v[..., 0], v[..., 1], color='k', zorder=5)
ax.scatter(vax[..., 0], vax[..., 1], color='k', marker='x', zorder=5)

#fig.savefig('bilat_45.svg', bbox_inches = 'tight')
fig.savefig('bilat_37.98.svg', bbox_inches = 'tight')
#fig.savefig('bilat_10.svg', bbox_inches = 'tight')
#%% 3
pt1 = np.array([-0.5,0.25,np.sqrt(0.6875)])

someplane = np.cross(pt1, northpole)
someplane /= np.linalg.norm(someplane)
startpt1 = rodrigues(someplane, pt1, 30*np.pi/180)

circle1 = rodrigues(pt1[np.newaxis], startpt1,
                    alltheta[:, np.newaxis])
pt2 = -pt1
pt2[2] = pt1[2]
startpt2 = rodrigues(someplane, pt2, 37.97*np.pi/180)#45*np.pi/180)10*np.pi/180)#

circle2 = rodrigues(pt2[np.newaxis], startpt2,
                    alltheta[:, np.newaxis])

pt3 = np.array([-0.4, -0.4, np.sqrt(0.68)])
startpt3 = rodrigues(someplane, pt3, 60*np.pi/180)
pts = np.stack([pt1, pt2, pt3])#clockwise but that's ok
cosrs = np.array([pt1 @ startpt1, pt2 @ startpt2, pt3 @ startpt3])
v3 = np.linalg.solve(pts, cosrs)
v3 /= np.linalg.norm(v3)

cxcx = -np.stack([np.cross(pt2, pt3), #negative to deal with clockwise
                 np.cross(pt3, pt1), 
                 np.cross(pt1, pt2)]).T
v = cxcx @ cosrs
v /= np.linalg.norm(v)

circle3 = rodrigues(pt3[np.newaxis], startpt3, alltheta[:, np.newaxis])
fig, ax = plotinit()
ax.scatter(pt1[0], pt1[1], color='r')
ax.plot(circle1[..., 0], circle1[..., 1], color='r')
ax.scatter(pt2[0], pt2[1], color='b')
ax.plot(circle2[..., 0], circle2[..., 1], color='b')
ax.scatter(pt3[0], pt3[1], color='g')
ax.plot(circle3[..., 0], circle3[..., 1], color='g')
ax.scatter(v[..., 0], v[..., 1], color='k', zorder=5)

fig.savefig('trilat_60.svg', bbox_inches = 'tight')
#%% 4
pt4 = np.array([0.4, 0.4, np.sqrt(0.68)])
startpt4 = rodrigues(someplane, pt4, 20*np.pi/180)
circle4 = rodrigues(pt4[np.newaxis], startpt4, alltheta[:, np.newaxis])

pts = np.stack([pt1, pt2, pt3, pt4])
cosrs = np.array([pt1 @ startpt1, pt2 @ startpt2, pt3 @ startpt3, pt4 @ startpt4])

#v1, _, _, _ = np.linalg.lstsq(pts, cosrs)
#v1 /= np.linalg.norm(v1)

ptp = pts.T @ pts
ptc = pts.T @ cosrs

pxpx = np.stack([np.cross(ptp[1], ptp[2]),
                  np.cross(ptp[2], ptp[0]),
                  np.cross(ptp[0], ptp[1])])
v = pxpx @ ptc 
v /= np.linalg.norm(v)

fig, ax = plotinit()
ax.scatter(pt1[0], pt1[1], color='r')
ax.plot(circle1[..., 0], circle1[..., 1], color='r', zorder=5)
ax.scatter(pt2[0], pt2[1], color='b')
ax.plot(circle2[..., 0], circle2[..., 1], color='b', zorder=5)
ax.scatter(pt3[0], pt3[1], color='g')
ax.plot(circle3[..., 0], circle3[..., 1], color='g', zorder=5)
ax.scatter(pt4[0], pt4[1], color='g')
ax.plot(circle4[..., 0], circle4[..., 1], color='y', zorder=5)

ax.scatter(v[..., 0], v[..., 1], color='k', zorder=5)

fig.savefig('multilat_20.svg', bbox_inches = 'tight')
