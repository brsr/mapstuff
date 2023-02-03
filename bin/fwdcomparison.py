#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:54:43 2020

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

#import os
#os.chdir('Code/mapstuff')
import mapstuff

#geod = pyproj.Geod(a=1, b=1)
geod = pyproj.Geod(a=6371, f=0)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#tetrahedron face
# tetv = np.array([[-np.sqrt(8),          0, -1],
#                  [ np.sqrt(2), np.sqrt(6), -1],
#                  [ np.sqrt(2),-np.sqrt(6), -1],
#                  [          0,          0,  3]]).T/3
# theta = -1/18*np.pi
# mat = np.array([[ np.cos(theta), 0, np.sin(theta)],
#                 [             0, 1,             0],
#                 [-np.sin(theta), 0, np.cos(theta)]])

# tetv = mat.T @ tetv
# tetll = mapstuff.UnitVector.invtransform_v(tetv)
# tetll += np.array([30, 0])[:,np.newaxis]
# actrlpts = tetll[:, 1:]
# actrlpts = actrlpts[:, ::-1]
# actrlpts = np.array([[ 30.        , -33.54104635,  93.54104635],
#                      [ 80.        , -24.21286243, -24.21286243]])
#octahedron face
#octll = np.array([[0,-45,45],
#                  [90,-0,0]])
#ov = mapstuff.UnitVector.transform_v(octll)
#print(np.pi/2*6371)
#actrlpts = np.array([[ 0.        , -20.22,  70.21857],
#                     [ 80.        , -5, -5]])
#theta = -1/18*np.pi
#mat = np.array([[ np.cos(theta), 0, np.sin(theta)],
#                [             0, 1,             0],
#                [-np.sin(theta), 0, np.cos(theta)]])

#ov = mat.T @ ov
#actrlpts = mapstuff.UnitVector.invtransform_v(ov)
#actrlpts += np.array([20,0])[:,np.newaxis]

#icosahedron face
a = np.arctan(1/2)/np.pi*180
actrlpts3 = np.array([[15+0, 15+36, 15-36],
                [-a, a, a]])
#iv = mapstuff.UnitVector.transform_v(ill)
#theta = 0.29*np.pi
#mat = np.array([[ np.cos(theta), 0, np.sin(theta)],
#                [             0, 1,             0],
#                [-np.sin(theta), 0, np.cos(theta)]])

#iv = mat.T @ iv
#actrlpts3 = mapstuff.UnitVector.invtransform_v(iv)
#actrlpts3 += np.array([15,0])[:,np.newaxis]

#actrlpts3 = np.array([[0,    0, 179.999],
#                     [60, -60, 0]])
#actrlpts3 = np.array([[-20,    -20, 80],
#                     [45,  -45, 0]])

ctrlpoly3 = mapstuff.geodesics(actrlpts3[0], actrlpts3[1], geod, includepts=True)
a = 180/np.pi * np.arctan(1/np.sqrt(2))
actrlpts4 = np.array([[-30, 60, 60, -30],
                      [-a, -a, a, a]])
ctrlpoly4 = mapstuff.geodesics(actrlpts4[0], actrlpts4[1], geod, includepts=True)
ctrlpoly3.crs = world.crs
ctrlpoly4.crs = world.crs

#controlpts = mapstuff.arraytoptseries(actrlpts)#, crs=ucrs)
vcontrolpts3 = mapstuff.UnitVector.transform_v(actrlpts3)
vcontrolpts4 = mapstuff.UnitVector.transform_v(actrlpts4)

center3 = mapstuff.UnitVector.invtransform_v(vcontrolpts3.sum(axis=1))[:,np.newaxis]
small3 = np.concatenate([center3, actrlpts3[:,1:]], axis=1)
center4 = np.array([15, 0])[:,np.newaxis]
small4 = np.concatenate([center4, actrlpts4[:,:2]], axis=1)
vsmall3 = mapstuff.UnitVector.transform_v(small3)
vsmall4 = mapstuff.UnitVector.transform_v(small4)

#cx = np.cross(vcontrolpts4, np.roll(vcontrolpts4, -1, axis=1), axis=0)

#ctrlboundary = mapstuff.geodesics(actrlpts[0], actrlpts[1], geod)
#ctrlpoly = geopandas.GeoSeries(pd.concat([ctrlboundary, controlpts],
#                                            ignore_index=True),
#                               crs=world.crs)

adegpts = np.array(np.meshgrid(np.linspace(-180, 180, 361),
                               np.linspace(-90, 90, 181)))
#adegpts = np.array(np.meshgrid(np.arange(-45,106),
#                               np.arange( -60, 90)))
#adegpts = np.array(np.meshgrid(np.linspace(-45, 105, 301),
#                               np.linspace(-57, 89.5, 294)))
#adegpts = np.array(np.meshgrid(np.linspace(-34, 94, 257),
#                               np.linspace(-46, 80, 253)))
adegpts = np.array(np.meshgrid(np.linspace(-30, 55, 426),#171),
                               np.linspace(15, 80, 326)))#131))
adegpts = np.array(np.meshgrid(np.linspace(-45, 75, 121),
                               np.linspace(-60, 60, 121)))
vdegpts = mapstuff.UnitVector.transform_v(adegpts)
#insidemask3 = np.all(np.tensordot(np.linalg.inv(vcontrolpts3), vdegpts,
insidemask3 = np.all(np.tensordot(np.linalg.inv(vsmall3), vdegpts,
                                 axes=(1, 0)) > 0, axis=0)
#cx2 = np.einsum('ij,i...', cx, vdegpts)
#insidemask4 = np.all(cx2 > 0, axis=-1)
insidemask4 = np.all(np.tensordot(np.linalg.inv(vsmall4), vdegpts,
                                 axes=(1, 0)) > 0, axis=0)

degpts = mapstuff.arraytoptseries(adegpts)#, crs=ucrs)
degpts.crs = world.crs
grat = mapstuff.graticule()
grat = mapstuff.graticule(lonrange = [-30, 60], latrange = [15, 90])
grat = mapstuff.graticule(lonrange = [0, 180], latrange = [-90, 90])
grat = mapstuff.graticule(lonrange = [-45, 75], latrange = [-60, 60])

center3 = mapstuff.UnitVector.invtransform_v(vcontrolpts3.sum(axis=1))
center4 = np.array([15, 0])
#midpoints = mapstuff.UnitVector.invtransform_v(vcontrolpts3 +
#                                              np.roll(vcontrolpts3, 1, axis=1))
#apoi = np.concatenate([actrlpts, midpoints, center[:, np.newaxis]], axis=1)
#poi = mapstuff.arraytoptseries(apoi)#, crs=ucrs)
fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='k')
grat.plot(ax=ax, color='lightgrey')
ctrlpoly3.plot(ax=ax, color='g')
ctrlpoly4.plot(ax=ax, color='g')
#poi.plot(ax=ax, color='y')
ax.axis('equal')

cycle = [0,1,2,0]
sidelengths = geod.line_lengths(actrlpts3[0][cycle], actrlpts3[1][cycle])
print(sidelengths)
tgtpts3 = mapstuff.trigivenlengths(sidelengths)#[::-1,]
ctrlarea3, _ = geod.polygon_area_perimeter(actrlpts3[0],
                                          actrlpts3[1])
scalefactor = np.sqrt(ctrlarea3/mapstuff.shoelace(tgtpts3))
mat = np.array([[0,-1],
                [1,0]]).T
tgtpts3 = mat @ tgtpts3 * scalefactor
tgtpts3 = tgtpts3[:,[2,0,1]]
ctrlarea4, _ = geod.polygon_area_perimeter(actrlpts4[0],
                                           actrlpts4[1])
cycle = [0,1,2,3,0]
sidelengths = geod.line_lengths(actrlpts4[0][cycle], actrlpts4[1][cycle])
print(sidelengths)

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
worlds = {}
grats = {}
ctrlpolys = {}
#pois = {}
degptss = {}
nctrlpts = {}
#proj projections
crs3= {'proj': 'gnom',
      'lat_0': center3[1],
      'lon_0': center3[0]}

crs4= {'proj': 'gnom',
      'lat_0': center4[1],
      'lon_0': center4[0]}

for i, crs, ctrlpoly in zip([3, 4], [crs3, crs4], [ctrlpoly3, ctrlpoly4]):
    name = 'Gnomonic ' + str(i)
    print(name)
    worlds[name]    = world.geometry.to_crs(crs)
    grats[name]     = grat.to_crs(crs)
    cp              = ctrlpoly.to_crs(crs)
    ctrlpolys[name] = cp
    nctrlpts[name] = i
    #pois      = poi.to_crs(crs)
    dp = mapstuff.ptseriestoarray(degpts.to_crs(crs)).reshape(adegpts.shape)
    if i == 3:
        gscale = cp[5].xy[1][0]/tgtpts3[1,2]
    elif i == 4:
        gscale = cp[7].xy[1][0]*2
    #gscale3 = pois[name][0].xy[1]/tgtpts3[1, 0]
    #dp = degptss[name]
    degptss[name] = dp/gscale

    def transform_gscale(x, y):
        return x/gscale, y/gscale
    worlds[name] = mapstuff.transeach(transform_gscale, worlds[name])
    grats[name] = mapstuff.transeach(transform_gscale, grats[name])
    ctrlpolys[name] = mapstuff.transeach(transform_gscale, ctrlpolys[name])
    #pois[name] = mapstuff.transeach(transform_gscale, pois[name])

#not-barycentric projections implemented here
projs = {#'Chamberlin Trimetric': mapstuff.ChambTrimetric(actrlpts, geod),#not polygonal
         #'Linear Trimetric':     mapstuff.LinearTrimetric(actrlpts, geod),#not polygonal
         #'Conformal2': mapstuff.ConformalTri(actrlpts, tgtpts),#buggy
         'Conformal': mapstuff.ConformalTri3(actrlpts3, tgtpts3),
         'Crider':              mapstuff.CriderEq(actrlpts4),
         'Snyder Equal-Area 4':  mapstuff.SnyderEA4(actrlpts4)
         }

for name in projs:
    print(name)
    mp = projs[name]
    nctrlpts[name] = mp.nctrlpts
    worlds[name] = mapstuff.transeach(mp.transform, world.geometry)
    grats[name] = mapstuff.transeach(mp.transform, grat)
    if mp.nctrlpts == 3:
        ctrlpolys[name] = mapstuff.transeach(mp.transform, ctrlpoly3)
    elif mp.nctrlpts == 4:
        ctrlpolys[name] = mapstuff.transeach(mp.transform, ctrlpoly4)
    #pois[name] = mapstuff.transeach(mp.transform, poi)
    degptss[name] = mp.transform_v(adegpts)

#barycentric projections
baryprojs = {'Areal':                   mapstuff.Areal(actrlpts3),
             'Fuller explicit':         mapstuff.FullerEq(actrlpts3),
             #'Fuller':                  mapstuff.FullerTri(actrlpts3, tweak=False),
             #'Fuller Tweaked':          mapstuff.FullerTri(actrlpts3, tweak=True),
             'Bisect':                  mapstuff.BisectTri(actrlpts3),
             'Bisect2':                  mapstuff.BisectTri2(actrlpts3),
             #'Snyder Equal-Area':       mapstuff.SnyderEA(actrlpts3),#not symmetric
             'Snyder Equal-Area 3':     mapstuff.SnyderEA3(actrlpts3),
             'Snyder Symmetrized':      mapstuff.SnyderEASym(actrlpts3),
             #'Alfredo':         mapstuff.Alfredo(actrlpts3),#polygonal?
             #'Alfredo Tweaked': mapstuff.Alfredo(actrlpts3, tweak=True),#not polygonal
             'Double':                  mapstuff.Double(actrlpts3, 
                                                       mapstuff.FullerEq, 
                                                       mapstuff.Areal, t=7)
             }

bp = mapstuff.Barycentric(tgtpts3)

worldsb    = {}
gratsb     = {}
ctrlpolysb = {}
#poisb      = {}
degptssb   = {}
for name in baryprojs:
    print(name)
    mp = baryprojs[name]
    nctrlpts[name] = mp.nctrlpts
    world_b = mapstuff.transeach(mp.transform, world.geometry)
    grat_b = mapstuff.transeach(mp.transform, grat)
    ctrlpoly_b = mapstuff.transeach(mp.transform, ctrlpoly3)
    #poi_b = mapstuff.transeach(mp.transform, poi)
    degpts_b = mp.transform_v(adegpts)

    worldsb[name] = world_b
    gratsb[name] = grat_b
    ctrlpolysb[name] = ctrlpoly_b
    #poisb[name] = poi_b
    degptssb[name] = degpts_b

    worlds[name] = mapstuff.transeach(bp.transform, world_b)
    grats[name] = mapstuff.transeach(bp.transform, grat_b)
    ctrlpolys[name] = mapstuff.transeach(bp.transform, ctrlpoly_b)
    #pois[name] = mapstuff.transeach(bp.transform, poi_b)
    degptss[name] = bp.transform_v(degpts_b)
    
baryprojs2 = {'SEA3':     mapstuff.SnyderEA(small3),
              'SEA4':     mapstuff.SnyderEA(small4)}

tgtsmall3 = tgtpts3.copy()
tgtsmall3[:,0] = 0
tgtsmall4 = np.array([[0.5, 0, 1],
                      [0.5, 0, 0]])
bp2 = {'SEA3':     mapstuff.Barycentric(tgtsmall3),
       'SEA4':     mapstuff.Barycentric(tgtsmall4)}
nctrlpts['SEA3'] = 3
nctrlpts['SEA4'] = 4
for name in baryprojs2:
    print(name)
    mp = baryprojs2[name]
    #nctrlpts[name] = mp.nctrlpts
    world_b = mapstuff.transeach(mp.transform, world.geometry)
    grat_b = mapstuff.transeach(mp.transform, grat)
    if nctrlpts[name] == 3:
        cpoly = ctrlpoly3
    else:
        cpoly = ctrlpoly4
    ctrlpoly_b = mapstuff.transeach(mp.transform, cpoly)
    #poi_b = mapstuff.transeach(mp.transform, poi)
    degpts_b = mp.transform_v(adegpts)

    worldsb[name] = world_b
    gratsb[name] = grat_b
    ctrlpolysb[name] = ctrlpoly_b
    #poisb[name] = poi_b
    degptssb[name] = degpts_b
    
    b = bp2[name]
    worlds[name] = mapstuff.transeach(b.transform, world_b)
    grats[name] = mapstuff.transeach(b.transform, grat_b)
    ctrlpolys[name] = mapstuff.transeach(b.transform, ctrlpoly_b)
    #pois[name] = mapstuff.transeach(b.transform, poi_b)
    degptss[name] = b.transform_v(degpts_b)
    
#%% remove interpolations
newdict = {}
#    {k:v for (k, v) in a.items() if any(k.startswith(k2) for k2 in b)}
for key, value in degptss.items():
    if not '%' in key:
        newdict[key] = value
degptss = newdict
#%% interpolated
name1 = 'Areal'
degpts1 = degptss[name1]

name2 = 'Gnomonic 3'
degpts2 = degptss[name2]

intervals = np.linspace(-0.4, 0, 41)
for t in intervals:
    newname = str(round((1-t)*100)) + '% ' + name1 + ', ' + str(round(t*100)) + '% ' + name2
    degpts = (1-t)*degpts1 + t*degpts2
    degptss[newname] = degpts
    nctrlpts[newname] = nctrlpts[name1]

name1 = 'Fuller explicit'
degpts_b1 = degptssb[name1]

name2 = 'Areal'
degpts_b2 = degptssb[name2]

intervals = np.linspace(1, 7, 29)
for t in intervals:
    newname = str(round((1-t)*100)) + '% ' + name1 + ', ' + str(round(t*100)) + '% ' + name2
    degpts_b = (1-t)*degpts_b1 + t*degpts_b2
    degptssb[newname] = degpts_b
    degptss[newname] = bp.transform_v(degpts_b)
    nctrlpts[newname] = nctrlpts[name1]

name1 = 'Gnomonic 3'
degpts1 = degptss[name1]

name2 = 'Conformal'
degpts2 = degptss[name2]

intervals = np.linspace(0, 1, 11)
for t in intervals:
    newname = str(round((1-t)*100)) + '% ' + name1 + ', ' + str(round(t*100)) + '% ' + name2
    degpts = (1-t)*degpts1 + t*degpts2
    degptss[newname] = degpts
    nctrlpts[newname] = nctrlpts[name1]    

#%% calculate metrics
#anglemax = max(projs['Conformal'].ctrl_angles - 60)

scales = {}
angles = {}
#distances = {}
#pctdistances = {}
for name in degptss:
    degpts_t = degptss[name]
    n = nctrlpts[name]
    omega, scale = mapstuff.omegascale(adegpts, degpts_t,
                                      geod, spacing=1)
    # if name == 'Gnomonic 3':
    #     scalefactor = 1
    # elif name == 'Gnomonic 4':
    #     scalefactor = 1
    if n == 3:
        scalefactor = 1
    elif n == 4:
        scalefactor = ctrlarea4
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
    #distances[name] = dist[0]
    #pctdistances[name] = pctdist[0]

#%% aggregate stats
ms = ['Scale', 'Angle']#, 'Distance', 'Pct Distance']
projnames = degptss.keys()
#projnames = ['Conformal', 'Gnomonic', 'Bisect', 'Areal', 'Fuller explicit',
#             'Fuller',
#       'Fuller Tweaked', 'Snyder Symmetrized', 'Snyder Equal-Area',
#       'Alfredo', 'Alfredo Tweaked', 'SEA']
index = pd.MultiIndex.from_product([projnames, ms],
                                   names=['Projection', 'Measure'])
cols = ['min', 'max', 'ave', 'stdev']#'q1', 'q99',
dat = pd.DataFrame(index = index, columns=cols)

for measures, mname in zip([scales, angles], ms):#, distances, pctdistances
    for name in measures:
        m = measures[name]
        n = nctrlpts[name]
        #minima
        maskm = m.copy()
        if n == 3:
            insidemask = insidemask3
        elif n == 4:
            insidemask = insidemask4

        maskm[~insidemask] = np.nan
        #weight so the average comes out right
        wght = np.cos(adegpts[1]*np.pi/180)
        wght[~insidemask] = np.nan
        avgm = np.nansum(maskm*wght)/np.nansum(wght)
        sqdif = (maskm - avgm)**2
        stdev = np.sqrt(np.nansum(sqdif*wght)/np.nansum(wght))
        mq = np.quantile(maskm[np.isfinite(maskm)],
                         [0, 1])#0.01, 0.99, 1])
        dat.loc[name,mname] = list(mq) + [avgm, stdev]

#dat.loc['Conformal', 'Angle'] = 0
#dat.loc['Conformal 3', 'Scale']['max'] = np.inf
#dat.loc['Conformal 3', 'Angle']['max'] = anglemax
#dat.loc['Snyder Equal-Area 3', 'Scale'] = 1.0
#dat.loc['Snyder Equal-Area 3', 'Scale']['stdev'] = 0.0
#dat.loc['Snyder Equal-Area 4', 'Scale'] = 1.0
#dat.loc['Snyder Equal-Area 4', 'Scale']['stdev'] = 0.0

#dat = dat.astype(float)
#print(dat)
#dat.to_csv('docs/Summary stats.csv')

#for name, scale in scales.items():
#    s = dat.xs([name,'Scale']).ave
#    scale /= s
#%%
angleave = dat.xs('Angle', level=1).ave
scalerat = dat.xs('Scale', level=1)['max']/dat.xs('Scale', level=1)['min']
#scalerat['Conformal'] = np.inf
#scalerat = dat.xs('Scale', level=1).stdev
#scalerat = dat.xs('Scale', level=1)['q99']/dat.xs('Scale', level=1)['q1']
#distave = dat.xs('Pct Distance', level=1)['max']#.ave#ratio

def log_10_product(x, pos):
    return '%1i' % (x)
tk = matplotlib.ticker.FuncFormatter(log_10_product)
index = [0,2] + list(range(5, 13))
tk = matplotlib.ticker.FuncFormatter(log_10_product)
fig, (ax1, ax2) = plt.subplots(nrows=2,  figsize=(8.5, 8))
ax1.scatter(scalerat[index], angleave[index])
ax1.plot(scalerat[14:], angleave[14:])
ax1.set_xlabel('Scale')
ax1.set_ylabel('Angle')
ax1.set_xscale('log')
ax1.set_ylim(0)
ax1.xaxis.set_major_formatter(tk)
for n, x, y in zip(angleave.index[index], scalerat[index], angleave[index]):
    if '%' not in n:
        ax1.annotate(n, (x, y), ha='center', va='bottom')
index = [1, 3, 4, 13]
ax2.scatter(scalerat[index], angleave[index])
ax2.set_xlabel('Scale')
ax2.set_ylabel('Angle')
ax2.set_xscale('log')
ax2.set_ylim(0)
ax2.xaxis.set_major_formatter(tk)
for n, x, y in zip(angleave.index[index], scalerat[index], angleave[index]):
    ax2.annotate(n, (x, y), ha='center', va='bottom')

fig.savefig('docs/Summary stats.svg')
#%%
scalefmts = {'Scale':  {'Gnomonic 3': '%1.1f',
                        'Gnomonic 4': '%1.1f',
                       'Conformal 3': '%1.1f',
                       'Conformal2': '%1.1f',
                       'Bisect': '%1.2f',
                       'Bisect2': '%1.2f',
                       'Double': '%1.1f',
                       'Areal': '%1.2f',
                       'Fuller explicit': '%1.2f',
                       'Fuller': '%1.2f',
                       'Fuller Tweaked': '%1.2f',
                       'Crider': '%1.2f',
                       'Snyder Symmetrized': '%1.3f',
                       'Snyder Equal-Area 3': '%1.3f',
                       'Snyder Equal-Area 4': '%1.3f',
                       'Alfredo': '%1.3f',
                       'Alfredo Tweaked': '%1.3f',
                       'SEA3': '%1.3f',
                       'SEA4': '%1.3f',                       
                       },
            'Angle':    '%1.0f',
            'Distance': '%1.0f',
            'Pct Distance': '%1.3f'}
ptfmts = {'Scale':    '%1.3f',
        'Angle':    '%1.1f',
        'Distance': '%1.0f',
        'Pct Distance':    '%1.3f'}
levels = {'Scale':    {'Gnomonic 3':          np.linspace(0.9, 1.5, 7),
                       'Gnomonic 4':          np.linspace(0.6, 2.6, 6),
                       'Conformal 3':         np.linspace(0.9, 1.5, 7),
                       'Conformal2':         np.linspace(0.9, 1.5, 7),
                       'Bisect':            np.linspace(0.95, 1.15, 5),
                       'Bisect2':            np.linspace(0.95, 1.15, 5),
                       'Double':            np.linspace(0.9, 1.5, 7),
                       'Areal':             np.linspace(0.95, 1.15, 5),
                       'Fuller explicit':   np.linspace(0.975, 1.075, 5),
                       'Fuller':    np.linspace(0.95, 1.05, 5),
                       'Fuller Tweaked':    np.linspace(0.95, 1.05, 5),
                       'Crider':    np.linspace(0.75, 1.5, 7),
                       'Snyder Symmetrized': np.linspace(1, 1.003, 4),
                       'Snyder Equal-Area 3': None,
                       'Snyder Equal-Area 4': None,
                       'Alfredo': None,
                       'Alfredo Tweaked': None,
                       'SEA3': np.linspace(0.95, 1.01, 7),                       
                       'SEA4': np.linspace(0.95, 1.01, 7)                       },
          'Angle':    {'Gnomonic 3':       np.linspace(0, 12, num=7),
                       'Gnomonic 4':          np.linspace(0, 30, num=7),
                       'Conformal 3':        np.linspace(0, 12, num=7),
                       'Conformal2':        np.linspace(0, 12, num=7),
                       'Double':            np.linspace(0, 12, num=7),                       
                       'Bisect':            np.linspace(0, 12, num=7),
                       'Bisect2':            np.linspace(0, 12, num=7),
                       'Areal':             np.linspace(0, 12, num=7),
                       'Fuller explicit':   np.linspace(0, 12, num=7),
                       'Fuller':    np.linspace(0, 12, num=7),
                       'Fuller Tweaked':    np.linspace(0, 12, num=7),
                       'Crider':    np.linspace(0, 30, num=7),
                       'Snyder Symmetrized': np.linspace(0, 12, num=7),
                       'Snyder Equal-Area 3': np.linspace(0, 12, num=7),
                       'Snyder Equal-Area 4': np.linspace(0, 30, num=7),
                       'Alfredo': None,
                       'Alfredo Tweaked': None,
                       'SEA3': np.linspace(0, 18, num=7),
                       'SEA4': np.linspace(0, 30, num=7)  },
          # 'Distance': np.linspace(-350, 200, 12),
          # 'Pct Distance': {'Gnomonic 3':          np.linspace(-0.05, 0.3, 8),
          #                  'Gnomonic 4':          np.linspace(-0.05, 0.3, 8),
          #                  'Conformal':         np.linspace(-0.05, 0.3, 8),
          #                  'Conformal2':         np.linspace(-0.05, 0.3, 8),
          #                  'Bisect':            np.linspace(-0.05, 0.1, 7),
          #                  'Areal':             np.linspace(-0.05, 0.1, 7),
          #                  'Fuller':   np.linspace(-0.05, 0.05, 5),
          #                  'Fuller explicit':   np.linspace(-0.05, 0.05, 5),
          #                  'Fuller Tweaked':    np.linspace(-0.05, 0.05, 5),
          #                  'Crider':    np.linspace(-0.05, 0.05, 5),
          #                  'Snyder Symmetrized': np.linspace(-0.05, 0.05, 5),
          #                  'Snyder Equal-Area 3': np.linspace(-0.05, 0.01, 7),
          #                  'Snyder Equal-Area 4': np.linspace(-0.05, 0.01, 7),
          #                  'Alfredo': None,
          #                  'Alfredo Tweaked': None,
          #              'SEA': None}
          }

for name in ['SEA3', 'SEA4']:#worlds:
    print(name)
    #mp = projs[name]
    world_t = worlds[name]
    grat_t = grats[name]
    ctrlpoly_t = ctrlpolys[name]
    degpts_t = degptss[name]
    n = nctrlpts[name]
    if n == 3:
        ctrlpoly = ctrlpoly3
    elif n == 4:
        ctrlpoly = ctrlpoly4

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8.5, 5), constrained_layout=True)
    fig.suptitle(name)
    ctrlpoly.plot(ax=ax1, color='g')
    grat.plot(ax=ax1, color='lightgrey')
    world.plot(ax=ax1, color='k')

    cb = ctrlpoly.bounds
    xlim1 = np.array([cb.minx.min() - 5,
                      cb.maxx.max() + 5])
    ylim1 = np.array([cb.miny.min() - 5,
                      cb.maxy.max() + 5])

    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)

    #2
    cbt = ctrlpoly_t.bounds
    if n == 4:
        xlim2 = np.array([cbt.minx.min() - 0.05,
                          cbt.maxx.max() + 0.05])
        ylim2 = np.array([cbt.miny.min() - 0.05,
                          cbt.maxy.max() + 0.05])
        if ylim2[1] < 0.55:
            ylim2[1] = 0.55        
        xlim2 = np.clip(xlim2, -0.05, 1.05)
        ylim2 = np.clip(ylim2, -0.05, 1.05)
    else:
        xlim2 = np.array([cbt.minx.min() * 1.1,
                          cbt.maxy.max() * 1.1])
        ylim2 = np.array([cbt.miny.min() * 1.1,
                          cbt.maxy.max() * 1.1])
        if ylim2[0] > -100:
            ylim2[0] = -100
        xlim2 = np.clip(xlim2, tgtpts3[0].min() * 1.1, tgtpts3[0].max() * 1.1)
        ylim2 = np.clip(ylim2, tgtpts3[1].min() * 1.1, tgtpts3[1].max() * 1.1)
        
        

    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)

    ctrlpoly_t.plot(ax=ax2, color='g')
    grat_t.plot(ax=ax2, color='lightgrey')
    world_t.plot(ax=ax2, color='k')
    fig.savefig('docs/' + name + '.svg')

    measures = {'Scale': scales[name],
                'Angle': angles[name],
                #'Distance': distances[name],
                #'Pct Distance': pctdistances[name]
                }
    for n in measures:
        print(n)
        m = measures[n]
        if m.std() == 0:
            continue

        scalefmt = scalefmts[n]
        ptfmt = ptfmts[n]
        level = levels[n]
        try:
            scalefmt = scalefmt[name]
        except:
            pass
        try:
            level = level[name]
        except:
            pass
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
        ax1.set_xlim(xlim1)
        ax1.set_ylim(ylim1)
        ax2.set_xlim(xlim2)
        ax2.set_ylim(ylim2)
        fig.suptitle(name + ': ' + n)
        fig.colorbar(cs)
        fig.savefig('docs/' + name + '_' + n + '.svg')