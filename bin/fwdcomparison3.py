#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:54:43 2020

@author: brsr
"""
import geopandas
import pandas as pd
#import shapely
#from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from scipy.spatial import ConvexHull
#import pickle

#import os
#os.chdir('Code/mapstuff')
import mapstuff

#geod = pyproj.Geod(a=1, b=1)
geod = pyproj.Geod(a=6371, f=0)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

#octahedron face
#octll = np.array([[0,-45,45],
#                  [90,-0,0]])
#ov = mapstuff.UnitVector.transform_v(octll)
#print(np.pi/2*6371)
#theta = -1/18*np.pi
#mat = np.array([[ np.cos(theta), 0, np.sin(theta)],
#                [             0, 1,             0],
#                [-np.sin(theta), 0, np.cos(theta)]])

#ov = mat.T @ ov
#actrlpts3 = mapstuff.UnitVector.invtransform_v(ov)
#actrlpts3 += np.array([20,0])[:,np.newaxis]

#actrlpts3 = np.array([[20., -25., 65.],
#                      [90-np.arctan(np.sqrt(2))*180/np.pi,0,0]])
                      #[90, 0, 0]])
actrlpts3 = mapstuff.multipart.icosa.sph_tri[7]
vcontrolpts3 = mapstuff.UnitVector.transform_v(actrlpts3)

#center3 = mapstuff.UnitVector.invtransform_v(vcontrolpts3.sum(axis=1))[:,np.newaxis]
#small3 = np.concatenate([center3, actrlpts3[:,1:]], axis=1)
#vsmall3 = mapstuff.UnitVector.transform_v(small3)

#actrlpts3 = small3

ctrlpoly3 = mapstuff.geodesics(actrlpts3[0], actrlpts3[1], geod, includepts=True)
ctrlpoly3.crs = world.crs

#adegpts = np.array(np.meshgrid(np.linspace(-180, 180, 360, endpoint=False),
#                               np.linspace(-90, 90, 180, endpoint=False)))+0.5
spacing = 0.5
adegpts = np.array(np.meshgrid(np.linspace(-45, 45, 181),
                               np.linspace(-45, 45, 181)))
vdegpts = mapstuff.UnitVector.transform_v(adegpts)
insidemask3 = np.all(np.tensordot(np.linalg.inv(vcontrolpts3), vdegpts,
#insidemask3 = np.all(np.tensordot(np.linalg.inv(vsmall3), vdegpts,
                                 axes=(1, 0)) > 0, axis=0)

degpts = mapstuff.arraytoptseries(adegpts)#, crs=ucrs)
degpts.crs = world.crs
#grat = mapstuff.graticule()
grat = mapstuff.graticule(lonrange = [-45, 45], latrange = [-45, 45])

center3 = mapstuff.UnitVector.invtransform_v(vcontrolpts3.sum(axis=1))
fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='k')
grat.plot(ax=ax, color='lightgrey')
ctrlpoly3.plot(ax=ax, color='g')
#poi.plot(ax=ax, color='y')
ax.axis('equal')

cycle = [0,1,2,0]
sidelengths = geod.line_lengths(actrlpts3[0][cycle], actrlpts3[1][cycle])
print(sidelengths)

tgtpts3 = mapstuff.multipart.icosa.tgt_tri[7]
#tgtpts3 = mapstuff.trigivenlengths(sidelengths)#[::-1,]
#angles = np.array([60,60,60])
#tgtpts3 = mapstuff.trigivenangles(angles)
#ctrlarea3, _ = geod.polygon_area_perimeter(actrlpts3[0],
#                                          actrlpts3[1])
#scalefactor = np.sqrt(ctrlarea3/mapstuff.shoelace(tgtpts3))
#tgtpts3 *= scalefactor
#mat = np.array([[0,-1],
#                [1,0]])
#tgtpts3 = mat @ tgtpts3 #* scalefactor
#tgtpts3 = tgtpts3[:,[2,0,1]]
#tgtpts3[1] -= tgtpts3[1,1]
#print(ctrlarea3, mapstuff.shoelace(tgtpts3))
#fig, ax = plt.subplots(figsize=(10, 5))
#ax.axis('equal')
#ax.plot(tgtpts3[0], tgtpts3[1])
#ct = mapstuff.float2d_to_complex(tgtpts.T).squeeze()
#ct = ct *1j*np.exp(-1j*np.angle(ct[0]))
#tgtpts = mapstuff.complex_to_float2d(ct).T

# def contours(param, levels=None, filled=True, label=True):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     if filled:
#         cs = ax.contourf(adegpts[0], adegpts[1], param, levels=levels)
#     else:
#         cs = ax.contour(adegpts[0], adegpts[1], param, levels=levels)
#     if label:
#         ax.clabel(cs)#, inline=1, fontsize=10, fmt='%1.0f')
#     #ctrlpoly.plot(ax=ax, color='g')
#     fig.colorbar(cs)
#     return fig, ax

#lengths = []
#for i in range(3):
#    f, b, l = geod.inv(adegpts[0], adegpts[1],
#                       actrlpts[0,i]*np.ones(adegpts.shape[1:]),
#                       actrlpts[1,i]*np.ones(adegpts.shape[1:]))
#    lengths.append(l)
#lengths = np.array(lengths)
#maskindex = ((adegpts[0] < -30) | (adegpts[0] > 60) |
#             (adegpts[1] < 0) | (adegpts[1] > 90) )
#scalefactor = geod.polygon_area_perimeter(actrlpts[0], actrlpts[1])[0]/mapstuff.shoelace(tgtpts)
#rpoi = np.round(apoi)
#ipoi = np.array(np.where(np.all(adegpts[:, np.newaxis] ==
#                                rpoi[...,np.newaxis,np.newaxis], axis=0))[1:])
#%%
#worlds = {}
grats = {}
ctrlpolys = {}
#pois = {}
degptss = {}
nctrlpts = {}
#proj projections

# projprojs = {'ISEA Proj':{'proj': 'isea',
#                           'orient': 'pole'}}

# for name in projprojs:
#     print(name)
#     crs = projprojs[name]
#     #worlds[name]    = world.geometry.to_crs(crs)
#     grats[name]     = grat.to_crs(crs).scale(1E-3, 1E-3)
#     ctrlpolys[name] = ctrlpoly3.to_crs(crs).scale(1E-3, 1E-3)
#     #pois      = poi.to_crs(crs)
#     dp = mapstuff.ptseriestoarray(degpts.to_crs(crs)).reshape(adegpts.shape)
#     degptss[name] = dp*1E-3

pSymSmallCircle = partial(mapstuff.SnyderEASym, p=mapstuff.SmallCircleEA)
icoprojs = {'Areal': mapstuff.IcosahedralProjection(mapstuff.Areal),
            'Trilinear': mapstuff.IcosahedralProjection(mapstuff.Trilinear),
            #'Split Area': mapstuff.IcosahedralProjection(mapstuff.SplitAreaTri),
            #'Split Length': mapstuff.IcosahedralProjection(mapstuff.SplitLengthTri),
            #'Split Angle': mapstuff.IcosahedralProjection(mapstuff.SplitAngleTri),
            #'Bisect': mapstuff.IcosahedralProjection(mapstuff.BisectTri),
            #'Bisect 2': mapstuff.IcosahedralProjection(mapstuff.BisectTri2),
            'Snyder Sym': mapstuff.IcosahedralProjection(mapstuff.SnyderEASym),
            'Small Circle Sym': mapstuff.IcosahedralProjection(pSymSmallCircle),
            'Gnomonic': mapstuff.IcosahedralProjection(mapstuff.TriGnomonic),
            'Fuller': mapstuff.IcosahedralProjection(mapstuff.FullerEq),
            'Conformal': mapstuff.IcosahedralProjectionNoPost(mapstuff.ConformalTri),
            'Snyder C': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA),
            'Snyder M': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA, first=1),
            'Snyder V': mapstuff.SubdividedIcosahedralProjection(mapstuff.SnyderEA, first=2),
            'Small Circle C': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA),
            'Small Circle M': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA, first=1),
            'Small Circle V': mapstuff.SubdividedIcosahedralProjection(mapstuff.SmallCircleEA, first=2),
             }
icoprojs['O Flation'] = mapstuff.Double(
    icoprojs['Small Circle Sym'],
    icoprojs['Snyder Sym'], t = 1.95)#2)
icoprojs['O Angle'] = mapstuff.Double(
    icoprojs['Areal'],
    icoprojs['Fuller'], t = -5.98)
icoprojs['O Cusps'] = mapstuff.Double(
    icoprojs['Fuller'],
    icoprojs['Snyder Sym'], t = 1.41)
plist = [icoprojs[x] for x in ['Snyder Sym',
                                'Small Circle Sym',
                                'Fuller',
                                #'Bisect 2',
                                'Areal',
                                'Trilinear',                                 
                                #'Bisect',
                                'Gnomonic'
                                ]]
opt_f = np.array([ 2.013, -1.021,  0, 0,  0, 0.008])
opt_a = np.array([ -1.332,  6.773,  11.567, -14.16 ,  -7.365, 5.517])
opt_c = np.array([ 1.377,  0.07 , -0.497,  0.159, -0.029, -0.080])
icoprojs['MO Flation'] = mapstuff.Multiple(plist, opt_f)
icoprojs['MO Angle'] = mapstuff.Multiple(plist, opt_a)
icoprojs['MO Cusps'] = mapstuff.Multiple(plist, opt_c)

plist = [icoprojs[x] for x in ['Snyder C',
                               'Snyder M', 
                               'Snyder V', 
                               'Small Circle C', 
                               'Small Circle M',
                               'Small Circle V' ]]
opt_c = np.array([ 0.002, -0.032,  1.167,  0.284, -0.492, 0.080])
icoprojs['MO EA Cusps'] = mapstuff.Multiple(plist, opt_c)

for name in icoprojs:
    print(name)
    mp = icoprojs[name]
    #worlds[name] = mapstuff.transeach(mp.transform, world.geometry)
    grats[name] = mapstuff.transeach(mp.transform, grat)
    ctrlpolys[name] = mapstuff.transeach(mp.transform, ctrlpoly3)
    #pois[name] = mapstuff.transeach(mp.transform, poi_b)
    degptss[name] = mp.transform_v(adegpts)

#pickle.dump(icoprojs, open('icoprojs.pickle', 'wb'))
#pickle.dump(grats, open('grats.pickle', 'wb'))
#pickle.dump(ctrlpolys, open('ctrlpolys.pickle', 'wb'))
#pickle.dump(degptss, open('degptss.pickle', 'wb'))

#%%
top = geod.npts(actrlpts3[0,0], actrlpts3[1,0],
                actrlpts3[0,2], actrlpts3[1,2], 1)[0]
bottom = actrlpts3[:,1]
center = center3
v = actrlpts3[:,2]
b2c = np.array(geod.npts(bottom[0], bottom[1],
                         center[0], center[1], npts=50)).T
c2t = np.array(geod.npts(center[0], center[1],
                         top[0], top[1], npts=50)).T
t2v = np.array(geod.npts(top[0], top[1],
                         v[0], v[1], npts=50)).T

#distance 1 meter instead of 1 km, let's see how that works
b2cx = np.array(geod.fwd(b2c[0], b2c[1], az=np.ones(b2c.shape[1])*90,
                dist=1E-3*np.ones(b2c.shape[1])))[:2]
c2tx = np.array(geod.fwd(c2t[0], c2t[1], az=np.ones(c2t.shape[1])*90,
                dist=1E-3*np.ones(c2t.shape[1])))[:2]
t2vandv = np.concatenate([t2v, v[:,np.newaxis]], axis=1)
pangle = 90 + geod.inv(t2vandv[0,:50], t2vandv[1,:50],
                       t2vandv[0,1:], t2vandv[1,1:])[0]
t2vx = np.array(geod.fwd(t2v[0], t2v[1], az=pangle,
                dist=1E-3*np.ones(c2t.shape[1])))[:2]

b2cpair = np.stack([b2c, b2cx], axis=-1)
c2tpair = np.stack([c2t, c2tx], axis=-1)
t2vpair = np.stack([t2v, t2vx], axis=-1)

e2e = np.stack([b2cpair, c2tpair, t2vpair], axis=-1)

e2es = {}

for name in icoprojs:
    print(name)
    mp = icoprojs[name]
    res = mp.transform_v(e2e.reshape(2, -1))
    e2es[name] = res.reshape(e2e.shape)

expected = np.array([90, 90, 180.])
cuspas = {}
cuspavgs = {}
for name in e2es:
    e2e_t = e2es[name]
    cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                             e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
    cuspas[name] = cp
    cpdiff = np.where(cp < 0, cp + 360, cp) - expected[np.newaxis]
    cuspavgs[name] = abs(cpdiff).max(axis=0)

cuspavg = pd.DataFrame(cuspavgs).T
cuspavg.columns = ['Vertex to Center', 'Midpoint to Center', 'Edge']
cuspavg.loc['Conformal'] = 0
cuspavg.round(3)
#%% calculate metrics
#anglemax = max(projs['Conformal'].ctrl_angles - 60)
scalefactor = 1

scales = {}
angles = {}
#mpdevs = {}
#extras_diff = {}
#distances = {}
#pctdistances = {}
n = 3
for name in degptss:
    degpts_t = degptss[name]
    #n = nctrlpts[name]
    omega, scale= mapstuff.omegascale(adegpts,
                                degpts_t, geod, spacing=spacing)#, extras= True)
    scale *= scalefactor
    scale[scale < 0] = np.nan
    #omega[insidemask & (omega > anglemax)] = anglemax
    #x = degpts_t[:,np.newaxis] - tgtpts[...,np.newaxis,np.newaxis]#?
    #y = np.sqrt(np.sum(x**2, axis=0))
    #results = y/lengths
    #dist = (y - lengths)#*180/np.pi
    #dist[~np.isfinite(dist)] = np.nan
    #pctdist = y/lengths - 1
    #pctdist[~np.isfinite(pctdist)] = np.nan
    scales[name] = scale
    angles[name] = omega
    #mpdevs[name] = mpdev
    #distances[name] = dist[0]
    #pctdistances[name] = pctdist[0]
#%% aggregate stats
ms = ['Flation', 'Angle']#, 'Distance', 'Pct Distance']
projnames = list(degptss.keys())# + ['Conformal']
index = pd.MultiIndex.from_product([projnames, ms],
                                   names=['Projection', 'Measure'])
cols = ['min', 'max', 'ave', 'stdev']#'q1', 'q99',
dat = pd.DataFrame(index = index, columns=cols)

insidemask = insidemask3
wght = np.cos(adegpts[1]*np.pi/180)
wght[~insidemask] = np.nan

mx = [scales, angles]#, mpdevs]
for measures, mname in zip(mx, ms):#, distances, pctdistances
    for name in measures:
        m = measures[name]
        #minima
        maskm = m.copy()
        maskm[~insidemask] = np.nan
        #weight so the average comes out right
        avgm = np.nansum(maskm*wght)/np.nansum(wght)
        sqdif = (maskm - avgm)**2
        stdev = np.sqrt(np.nansum(sqdif*wght)/np.nansum(wght))
        mq = np.quantile(maskm[np.isfinite(maskm)],
                         [0, 1])#0.01, 0.99, 1])
        dat.loc[name,mname] = list(mq) + [avgm, stdev]

#dat.loc['Conformal', 'Angle'] = 0
#dat.loc['Conformal', 'Flation'] = np.inf

#dat.loc['Conformal 3', 'Flation']['max'] = np.inf
#dat.loc['Conformal 3', 'Angle']['max'] = anglemax
#dat.loc['Snyder Equal-Area 3', 'Flation'] = 1.0
#dat.loc['Snyder Equal-Area 3', 'Flation']['stdev'] = 0.0
#dat.loc['Snyder Equal-Area 4', 'Flation'] = 1.0
#dat.loc['Snyder Equal-Area 4', 'Flation']['stdev'] = 0.0

#dat = dat.astype(float)
#print(dat)
#dat.to_csv('docs/Summary stats.csv')

#for name, scale in scales.items():
#    s = dat.xs([name,'Flation']).ave
#    scale /= s

#%% interpolated
pairs = [('Snyder Sym', 'Small Circle Sym'),
        ('Snyder Sym', 'Fuller'),
        ('Small Circle Sym', 'Fuller'),
        ('Snyder Sym', 'Bisect 2'),
        ('Small Circle Sym', 'Bisect 2'),
        ('Fuller', 'Bisect 2'),
        ('Snyder Sym', 'Areal'),
        ('Small Circle Sym', 'Areal'),
        ('Fuller', 'Areal'),
        ('Bisect 2', 'Areal'),
        ('Snyder Sym', 'Bisect'),
        ('Small Circle Sym', 'Bisect'),
        ('Fuller', 'Bisect'),
        ('Bisect 2', 'Bisect'),
        ('Areal', 'Bisect'),
        ('Snyder Sym', 'Gnomonic'),
        ('Small Circle Sym', 'Gnomonic'),
        ('Fuller', 'Gnomonic'),
        ('Bisect 2', 'Gnomonic'),
        ('Areal', 'Gnomonic'),
        ('Bisect', 'Gnomonic')
        ]
intervals = np.linspace(-7.5, 5, 50)
fratios = {}
avgomegas = {}
cuspsts = {}

for name1, name2 in pairs:
    newname = name1 + ' + ' + name2
    degpts1 = degptss[name1]
    degpts2 = degptss[name2]
    x = []
    y = []
    for t in intervals:
        degpts = t*degpts1 + (1-t)*degpts2
        omega, scale = mapstuff.omegascale(adegpts, degpts,
                                          geod, spacing=spacing)
        scale[~insidemask] = np.nan
        avgomega = np.nansum(omega*wght)/np.nansum(wght)
        fratio = np.nanmax(scale)/np.nanmin(scale)
        x.append(fratio)
        y.append(avgomega)
    fratios[newname] = x
    avgomegas[newname] = y

    newname = name1 + ' + ' + name2
    e2e1 = e2es[name1][..., 2]
    e2e2 = e2es[name2][..., 2]
    z = []
    for t in intervals:
        e2e_t = t*e2e1 + (1-t)*e2e2
        cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                                 e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
        cpdiff = np.where(cp < 0, cp + 360, cp) - expected[2]
        z.append(abs(cpdiff).max(axis=0))
    cuspsts[newname] = z

datcfr = pd.DataFrame(fratios, index=intervals)
datcomega = pd.DataFrame(avgomegas, index=intervals)
#%%
angleave = dat.xs('Angle', level=1).ave
scalerat = dat.xs('Flation', level=1)['max']/dat.xs('Flation', level=1)['min']
cuspy = cuspavg['Edge']

#%%
# plotpairs = [('Snyder Sym + Small Circle Sym'),
#              ('Snyder Sym + Fuller'),
#              ('Fuller + Areal')]


x = np.concatenate([scalerat.values.astype(float),
                    np.concatenate([x for x in fratios.values()])])
y = np.concatenate([angleave.values.astype(float),
                    np.concatenate([x for x in avgomegas.values()])])
z = np.concatenate([cuspy.values.astype(float),
                    np.concatenate([x for x in cuspsts.values()])])
points = np.array([x,y,z]).T
ind = np.any(np.signbit(points), axis=1) | (points[:,0] > 3)
points = points[~ind]
hull = ConvexHull(points[:,:2])

fig, ax = plt.subplots(figsize=(8.5, 8))
ax.scatter(scalerat, angleave)
ax.set_xlabel('Flation ratio')
ax.set_ylabel('Average max angle distortion')
ax.set_ylim(0, 10)
ax.set_xlim(1, 2.25)
for n, x, y in zip(angleave.index, scalerat, angleave):
    if '%' not in n:
        ax.annotate(n, (x, y), ha='center', va='bottom')
#for name in plotpairs:
#    x = fratios[name]
#    y = avgomegas[name]
#    ax.plot(x, y, label=name, alpha=0.5, c='k')
ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'k--', lw=2, alpha=0.5)
#fig.savefig('docs/Summary stats.svg')
#%%
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.scatter(scalerat, cuspy)
ax.set_xlabel('Flation ratio')
ax.set_ylabel('Max cusp angle')
ax.set_ylim(0, 13)
ax.set_xlim(1, 2.25)
for n, x, y in zip(angleave.index, scalerat, cuspy):
    if '%' not in n:
        ax.annotate(n, (x, y), ha='center', va='bottom')
#for name in plotpairs:
#    x = fratios[name]
#    z = cuspsts[name]
#    ax.plot(x, z, label=name, alpha=0.5, c='k')
#%%
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.scatter(angleave, cuspy)
ax.set_xlabel('Average max angle distortion')
ax.set_ylabel('Max cusp angle')
ax.set_ylim(0, 13)
ax.set_xlim(0, 10)
for n, x, y in zip(angleave.index, angleave, cuspy):
    if '%' not in n:
        ax.annotate(n, (x, y), ha='center', va='bottom')
#for name in plotpairs:
#    y = avgomegas[name]
#    z = cuspsts[name]
#    ax.plot(y, z, label=name, alpha=0.5, c='k')
#%%
scalefmts = {'Flation':  '%1.3f',
            'Angle':    '%1.0f',
            'Distance': '%1.0f',
            'Pct Distance': '%1.3f',
            'Deviation': '%1.0f'}
ptfmts = {'Flation':    '%1.3f',
        'Angle':    '%1.1f',
        'Distance': '%1.0f',
        'Pct Distance':    '%1.3f',
        'Deviation': '%1.1f'}
levels = {'Flation':  np.linspace(0.8, 1.8, 6),
          'Angle':    np.linspace(0, 12, num=7),
          'Deviation': np.linspace(0, 6, num=7),
          }

n = 3
for name in degptss:
    print(name)
    #mp = projs[name]
    #world_t = worlds[name]
    grat_t = grats[name]
    ctrlpoly_t = ctrlpolys[name]
    degpts_t = degptss[name]
    ctrlpoly = ctrlpoly3
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8.5, 5), constrained_layout=True)
    # fig.suptitle(name)
    # ctrlpoly.plot(ax=ax1, color='g')
    # grat.plot(ax=ax1, color='lightgrey')
    # world.plot(ax=ax1, color='k')

    # cb = ctrlpoly.bounds
    # xlim1 = np.array([cb.minx.min() - 5,
    #                   cb.maxx.max() + 5])
    # ylim1 = np.array([cb.miny.min() - 5,
    #                   cb.maxy.max() + 5])

    # ax1.set_xlim(xlim1)
    # ax1.set_ylim(ylim1)

    # #2
    # cbt = ctrlpoly_t.bounds
    # if n == 4:
    #     xlim2 = np.array([cbt.minx.min() - 0.05,
    #                       cbt.maxx.max() + 0.05])
    #     ylim2 = np.array([cbt.miny.min() - 0.05,
    #                       cbt.maxy.max() + 0.05])
    #     if ylim2[1] < 0.55:
    #         ylim2[1] = 0.55
    #     xlim2 = np.clip(xlim2, -0.05, 1.05)
    #     ylim2 = np.clip(ylim2, -0.05, 1.05)
    # else:
    #     xlim2 = np.array([cbt.minx.min() * 1.1,
    #                       cbt.maxy.max() * 1.1])
    #     ylim2 = np.array([cbt.miny.min() * 1.1,
    #                       cbt.maxy.max() * 1.1])
    #     if ylim2[0] > -100:
    #         ylim2[0] = -100
    #     xlim2 = np.clip(xlim2, tgtpts3[0].min() * 1.1, tgtpts3[0].max() * 1.1)
    #     ylim2 = np.clip(ylim2, tgtpts3[1].min() * 1.1, tgtpts3[1].max() * 1.1)



    # ax2.set_xlim(xlim2)
    # ax2.set_ylim(ylim2)

    # ctrlpoly_t.plot(ax=ax2, color='g')
    # grat_t.plot(ax=ax2, color='lightgrey')
    # world_t.plot(ax=ax2, color='k')
    # fig.savefig('docs/' + name + '.svg')

    measures = {'Flation': scales[name],
                'Angle': angles[name],
                #'Distance': distances[name],
                #'Pct Distance': pctdistances[name]
                #'Deviation': mpdevs[name]
                }
    for n in measures:
        print(n)
        m = measures[n]
        if m.std() == 0:
            continue

        scalefmt = scalefmts[n]
        ptfmt = ptfmts[n]
        level = levels[n]
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8.5, 5), constrained_layout=True)
        #1
        ctrlpoly.plot(ax=ax1, color='g')
        grat.plot(ax=ax1, color='lightgrey')
        cs = ax1.contour(adegpts[0], adegpts[1], m, levels=level)
        ax1.clabel(cs, fmt=scalefmt)

        #2
        ctrlpoly_t.plot(ax=ax2, color='g')
        grat_t.plot(ax=ax2, color='lightgrey')
        cs = ax2.contour(degpts_t[0], degpts_t[1], m, levels=level)
        ax2.clabel(cs, fmt=scalefmt)
        #ax1.set_xlim(xlim1)
        #ax1.set_ylim(ylim1)
        #ax2.set_xlim(xlim2)
        #ax2.set_ylim(ylim2)
        fig.suptitle(name + ': ' + n)
        fig.colorbar(cs)
        fig.savefig('docs/' + name + '_' + n + '.svg')
