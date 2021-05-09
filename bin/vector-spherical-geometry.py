#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:18:38 2021

@author: brsr
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mapproj import Areal, UnitVector, rodrigues, slerp, triangle_solid_angle

abc = np.zeros((3,3))
abc[0, :2] = [-0, -0.5]
abc[1, :2] = [ 0.5, -0.25]
abc[2, :2] = [-0.5, 0.5]
abc[:,2] = np.sqrt(1 - abc[:, 0]**2 - abc[:, 1]**2)
abc /= np.linalg.norm(abc, axis=1, keepdims=True)

detabc = np.linalg.det(abc)

midpoints = np.roll(abc, -1, axis=0) + np.roll(abc, 1, axis=0)
midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)

edgecenters = np.cross(np.roll(abc, -1, axis=0), np.roll(abc, 1, axis=0))
angs = np.sum(np.roll(edgecenters, -1, axis=0)* np.roll(edgecenters, 1, axis=0), axis=1)
ncx = np.linalg.norm(edgecenters, axis=1)
edgecenters /= np.linalg.norm(edgecenters, axis=1, keepdims=True)

bisectors = np.roll(edgecenters, -1, axis=0) - np.roll(edgecenters, 1, axis=0)
bisectors /= np.linalg.norm(bisectors, axis=1, keepdims=True)

dots = np.sum(np.roll(abc, -1, axis=0)* np.roll(abc, 1, axis=0), axis=1)

t = np.linspace(0,1)[:,np.newaxis,np.newaxis]
#abcedges = slerp(np.roll(abc, -1, axis=0), np.roll(abc, 1, axis=0), t)
alltheta = np.linspace(0, 2*np.pi, 360)

#this is easier than figuring out how to make broadcasting cooperate
abcedges = np.zeros((len(t),3,3))
for i in range(3):
    abcedges[:,i] = slerp(abc[i-1], abc[(i+1)%3], t).squeeze()

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

ab = abc[0] @ abc[1]
bc = abc[1] @ abc[2]
ca = abc[2] @ abc[0]
axb = np.cross(abc[0], abc[1])
bxc = np.cross(abc[1], abc[2])
cxa = np.cross(abc[2], abc[0])
naxb = np.linalg.norm(axb)
nbxc = np.linalg.norm(bxc)
ncxa = np.linalg.norm(cxa)
#%% circles
pt = np.array([-0.5,0.25,np.sqrt(0.6875)])
antipt = -pt
someplane = np.cross(pt, abc[0])
someplane /= np.linalg.norm(someplane)
angles = (np.array([45,90,135])*np.pi/180)[:,np.newaxis]
interpts = rodrigues(someplane, pt, angles)
circles = rodrigues(pt[np.newaxis, np.newaxis], interpts[np.newaxis],
                    alltheta[:, np.newaxis, np.newaxis])
fig, ax = plotinit()
ax.scatter(pt[0], pt[1], color='r')
ax.scatter(antipt[0], antipt[1], color='#FFBBBB')
for i in range(3):
    index = circles[..., i, 2] < 0
    cnan = circles[:, i, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c='r')
    index = circles[..., i, 2] > 0
    cnan = circles[:, i, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c='#FFBBBB', zorder=-1)

fig.savefig('circles.svg', bbox_inches = 'tight')
#%% intersection
d = np.arccos(edgecenters[1] @ edgecenters[2])
intersect1 = np.cross(edgecenters[1], edgecenters[2])
intersect1 /= np.linalg.norm(intersect1)
intersect2 = -intersect1
circle1 = rodrigues(edgecenters[1], intersect1, alltheta[:, np.newaxis])
circle2 = rodrigues(edgecenters[2], intersect1, alltheta[:, np.newaxis])
link = slerp(edgecenters[1], edgecenters[2], t)
fig, ax = plotinit()
ax.scatter(edgecenters[1, 0], edgecenters[1, 1], c='r')
ax.scatter(edgecenters[2, 0], edgecenters[2, 1], c='b')
ax.scatter(intersect1[0], intersect1[1], c='k', zorder=5)
ax.scatter(intersect2[0], intersect2[1], c='#BBBBBB', zorder=5)
ax.plot(link[..., 0], link[..., 1], 'k--')
ax.text(0.15, 0.75, '$\\theta$')
ax.text(-0.15, -0.5, '$\\theta$')
ax.text(-0.05, -0.4, '$\\pi - \\theta$')

for circle, color1, color2 in [(circle1, 'r', '#FFBBBB'),
                               (circle2, 'b', '#BBBBFF')]:
    index = circle[..., 2] < 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color1)
    index = circle[..., 2] > 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color2, zorder=-1)

fig.savefig('intersection.svg', bbox_inches = 'tight')
#%% fwd problem
pole = np.array([0,1,0])
pt = np.array([-0.25, 0.4, np.sqrt(1-0.25**2-0.4**2)])
#d1 = np.arccos(pt@pole)
cmeridian = np.cross(pt, pole)
cmeridian /= np.linalg.norm(cmeridian)
circle1 = rodrigues(cmeridian, pt, alltheta[:, np.newaxis])
angle = 108/180*np.pi
circle2 = rodrigues(pt, circle1, -angle)
center2 = rodrigues(pt, cmeridian, -angle)
endpt = rodrigues(center2, pt, 1)
d2 = np.arccos(pt@endpt)
link1 = slerp(pt, pole, t)
link2 = slerp(pt, endpt, t)
fig, ax = plotinit()
ax.scatter(pole[0], pole[1], c='k')
ax.scatter(pt[0], pt[1], c='k', zorder=5)
ax.scatter(endpt[0], endpt[1], c='k', zorder=5)
ax.scatter(cmeridian[0], cmeridian[1], c='#FFBBBB')#, zorder=5)
ax.scatter(center2[0], center2[1], c='#BBBBFF')#, zorder=5)
#ax.plot(link1[..., 0], link1[..., 1], c='r')
ax.plot(link2[..., 0], link2[..., 1], c='b')
ax.text(0, 1.02, '$\\mathbf{\\hat{n}}$')
ax.text(-0.23, 0.43, '$\\theta$')
ax.text(0.1, 0.15,'$\\ell$')
ax.text(-0.35, 0.27, '$\\mathbf{\\hat{a}}$')
ax.text(0.6, 0.01, '$\\mathbf{\\hat{b}}$')
ax.text(-0.95, 0, '$\\mathbf{\\hat{c}}_1$', c='#888888')
ax.text(0.32, 0.77, '$\\mathbf{\\hat{c}}_2$', c='#888888')


for circle, color1, color2, style in [(circle1, 'r', '#FFBBBB', '--'),
                                      (circle2, 'b', '#BBBBFF', '--')]:
    index = circle[..., 2] < 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color1, ls=style)
    index = circle[..., 2] > 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], ls=style, c=color2, zorder=-1)

fig.savefig('forward.svg', bbox_inches = 'tight')
#%% perpendicular through a point
circle1 = rodrigues(edgecenters[1], abc[0], alltheta[:, np.newaxis])
perpcenter = np.cross(edgecenters[1], abc[1])
perpcenter /= np.linalg.norm(perpcenter)
circle2 = rodrigues(perpcenter, abc[1], alltheta[:, np.newaxis])
intersect = np.cross(perpcenter, edgecenters[1])
intersect /= np.linalg.norm(intersect)
intersect2 = -intersect
fig, ax = plotinit()
ax.scatter(edgecenters[1, 0], edgecenters[1, 1], c='r', zorder=5)
ax.scatter(abc[1, 0], abc[1, 1], c='g', zorder=5)
ax.scatter(perpcenter[0], perpcenter[1], c='#BBBBFF')
ax.scatter(intersect[0], intersect[1], c='k', zorder=5)
ax.scatter(intersect2[0], intersect2[1], c='#BBBBBB', zorder=5)

ax.text(0.45, -0.18, '$\\mathbf{\\hat{a}}$')
ax.text(0.75, 0.43, '$\\mathbf{\\hat{c}}$')
ax.text(0.5, -0.6, '$\\mathbf{\\hat{g}}$', c='#888888')
ax.text(0.03, -0.54, '$\\mathbf{\\hat{x}}$')
ax.text(-0.15, 0.64, '$-\\mathbf{\\hat{x}}$', c='#888888')

for circle, color1, color2 in [(circle1, 'r', '#FFBBBB'),
                               (circle2, 'b', '#BBBBFF')]:
    index = circle[..., 2] < 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color1)
    index = circle[..., 2] > 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color2, zorder=-1)
print(perpcenter @ edgecenters[1])
fig.savefig('perpendicular.svg', bbox_inches = 'tight')
#%% triangle
circle1 = rodrigues(edgecenters[1], abc[0], alltheta[:, np.newaxis])
circle2 = rodrigues(edgecenters[2], abc[1], alltheta[:, np.newaxis])
circle3 = rodrigues(edgecenters[0], abc[2], alltheta[:, np.newaxis])

link = slerp(edgecenters[1], edgecenters[2], t)
fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], c='k', zorder=5)
ax.scatter(edgecenters[1, 0], edgecenters[1, 1], c='r')
ax.scatter(edgecenters[2, 0], edgecenters[2, 1], c='b')
ax.scatter(edgecenters[0, 0], edgecenters[0, 1], c='g')

for circle, color1, color2 in [(circle1, 'r', '#FFBBBB'),
                               (circle2, 'b', '#BBBBFF'),
                               (circle3, 'g', '#BBFFBB')]:
    index = circle[..., 2] < 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color1)
    index = circle[..., 2] > 0
    cnan = circle[:, :].copy()
    cnan[index, :] = np.nan
    ax.plot(cnan[..., 0], cnan[..., 1], c=color2, zorder=-1)
fig.savefig('triangle.svg', bbox_inches = 'tight')
#%% triangle centers
#%% centroid
zvm = abc.sum(axis=0)
zvm /= np.linalg.norm(zvm)

#this is easier than figuring out how to make broadcasting cooperate
vms = np.zeros((len(t),3,3))
for i in range(3):
    vms[:,i] = slerp(abc[i], midpoints[i], t).squeeze()

fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')
ax.scatter(midpoints[..., 0], midpoints[..., 1], color='b')
ax.plot(vms[..., 0], vms[..., 1], color='b')

ax.scatter(zvm[0], zvm[1], label='vertex-median', color='r', zorder=5)
ax.set_title('centroid (vertex-median)')
fig.savefig('vertex-median.svg', bbox_inches = 'tight')

#%% area centroid
pj = Areal(UnitVector.invtransform_v(abc.T))
llar = pj.invtransform(1/3,1/3,1/3)
zar = UnitVector.transform_v(llar)

vms = slerp(abc, zar, t)

fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')
ax.plot(vms[..., 0], vms[..., 1], color='b')

ax.scatter(zar[0], zar[1], label='centroid', color='r', zorder=5)
ax.set_title('centroid (area)')
fig.savefig('area_centroid.svg', bbox_inches = 'tight')
print(triangle_solid_angle(abc[0], abc[1], zar),
      triangle_solid_angle(abc[1], abc[2], zar),
      triangle_solid_angle(abc[2], abc[0], zar))
#%% incenter
zin = ncx @ abc
zin /= np.linalg.norm(zin)

bisectpt = np.cross(edgecenters, bisectors)
bisectpt /= np.linalg.norm(bisectpt, axis=1, keepdims=True)

#this is easier than figuring out how to make broadcasting cooperate
bps = np.zeros((len(t),3,3))
for i in range(3):
    bps[:,i] = slerp(abc[i], bisectpt[i], t).squeeze()

perps = np.cross(zin, edgecenters)
pint = np.cross(edgecenters, perps)
pint /= np.linalg.norm(pint, axis=1, keepdims=True)

theta = np.linspace(0, 2*np.pi, 360)[:,np.newaxis]
v = pint[0]
cxv = np.cross(zin, v)
cv = zin @ v
cc = v*np.cos(theta) + cxv*np.sin(theta) + zin*cv*(1-np.cos(theta))

fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')
ax.scatter(bisectpt[..., 0], bisectpt[..., 1], color='b')
ax.plot(bps[..., 0], bps[..., 1], color='b')
ax.scatter(zin[0], zin[1], label='incenter', color='r', zorder=5)
ax.plot(cc[..., 0], cc[..., 1], color='r')
ax.set_title('incenter')
print(pint @ zin)
fig.savefig('incenter.svg', bbox_inches = 'tight')
#%% circumcenter
zcr = np.sum(np.cross(np.roll(abc, -1, axis=0),
                      np.roll(abc, 1, axis=0)), axis=0)
denom = np.linalg.norm(zcr)
zcr /= denom
circumradius = detabc/denom

theta = np.linspace(0, 2*np.pi, 360)[:,np.newaxis]
v = abc[0]
cxv = np.cross(zcr, v)
cv = zcr @ v
cc = v*np.cos(theta) + cxv*np.sin(theta) + zcr*cv*(1-np.cos(theta))
fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')
ax.scatter(zcr[0], zcr[1], label='circumcenter', color='r', zorder=5)
ax.plot(cc[..., 0], cc[..., 1], color='b')
ax.set_title('circumcenter')
print(np.arccos(abc@zcr))
fig.savefig('circumcenter.svg', bbox_inches = 'tight')

#%% orthocenter
ab = abc[0] @ abc[1]
bc = abc[1] @ abc[2]
ca = abc[2] @ abc[0]
axb = np.cross(abc[0], abc[1])
bxc = np.cross(abc[1], abc[2])
cxa = np.cross(abc[2], abc[0])
zor = ca*bc*axb + ab*ca*bxc + ab*bc*cxa
zor /= np.linalg.norm(zor)

perps = np.cross(abc, edgecenters)
perps /= np.linalg.norm(perps, axis=1, keepdims=True)
perpint = np.cross(edgecenters, perps)
perpint /= np.linalg.norm(perpint, axis=1, keepdims=True)
dim = (len(t), 3, 3)
prs = np.zeros(dim)
prs2 = np.zeros(dim)
ext = np.zeros(dim)
for i in range(3):
    prs[:, i] = slerp(abc[i], perpint[i], t).squeeze()
    prs2[:, i] = slerp(abc[i], zor, t).squeeze()
    ext[:, i] = slerp(abc[i-2], perpint[i], t).squeeze()

fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')
ax.scatter(perpint[..., 0], perpint[..., 1], color='b')
ax.plot(prs[..., 0], prs[..., 1], color='b')
ax.plot(prs2[..., 0], prs2[..., 1], color='b')
ax.plot(ext[..., 0], ext[..., 1], color='k', linestyle='dashed')
ax.scatter(zor[0], zor[1], label='orthocenter', color='r', zorder=5)
ax.set_title('orthocenter')

fig.savefig('orthocenter.svg', bbox_inches = 'tight')

#%% all four
fig, ax = plotinit()
ax.scatter(abc[..., 0], abc[..., 1], color='k')
ax.plot(abcedges[..., 0], abcedges[..., 1], color='k')

ax.scatter(zvm[0], zvm[1], label='vertex-median')
ax.scatter(zar[0], zar[1], label='areal')
ax.scatter(zin[0], zin[1], label='incenter')
ax.scatter(zcr[0], zcr[1], label='circumcenter')
ax.scatter(zor[0], zor[1], label='orthocenter')
ax.legend()
ax.set_title('triangle centers')

print(zor @ np.cross(zvm, zcr))
print(zor @ np.cross(zar, zcr))

fig.savefig('centers.svg', bbox_inches = 'tight')
