# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:07:12 2023

@author: brsto
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
from scipy.optimize import minimize_scalar, minimize
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
spacing = 1
adegpts = np.array(np.meshgrid(np.linspace(-45, 45, 91),
                               np.linspace(-45, 45, 91)))
vdegpts = mapstuff.UnitVector.transform_v(adegpts)
insidemask3 = np.all(np.tensordot(np.linalg.inv(vcontrolpts3), vdegpts,
#insidemask3 = np.all(np.tensordot(np.linalg.inv(vsmall3), vdegpts,
                                 axes=(1, 0)) > 0, axis=0)
insidemask = insidemask3
wght = np.cos(adegpts[1]*np.pi/180)
wght[~insidemask] = np.nan

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
            'Bisect': mapstuff.IcosahedralProjection(mapstuff.BisectTri),
            'Bisect 2': mapstuff.IcosahedralProjection(mapstuff.BisectTri2),
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
for name in icoprojs:#
    print(name)
    mp = icoprojs[name]
    #worlds[name] = mapstuff.transeach(mp.transform, world.geometry)
    grats[name] = mapstuff.transeach(mp.transform, grat)
    ctrlpolys[name] = mapstuff.transeach(mp.transform, ctrlpoly3)
    #pois[name] = mapstuff.transeach(mp.transform, poi_b)
    degptss[name] = mp.transform_v(adegpts)
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
#%% interpolated flation
pairs = [('Snyder Sym', 'Small Circle Sym'),							
('Snyder Sym', 'Fuller'),	('Small Circle Sym', 'Fuller'),						
('Snyder Sym', 'Bisect 2'),	('Small Circle Sym', 'Bisect 2'),	('Fuller', 'Bisect 2'),					
('Snyder Sym', 'Areal'),	('Small Circle Sym', 'Areal'),	('Fuller', 'Areal'),	('Bisect 2', 'Areal'),				
('Snyder Sym', 'Trilinear'),	('Small Circle Sym', 'Trilinear'),	('Fuller', 'Trilinear'),	('Bisect 2', 'Trilinear'),	('Areal', 'Trilinear'),			
('Snyder Sym', 'Bisect'),	('Small Circle Sym', 'Bisect'),	('Fuller', 'Bisect'),	('Bisect 2', 'Bisect'),	('Areal', 'Bisect'),	('Trilinear', 'Bisect'),		
('Snyder Sym', 'Gnomonic'),	('Small Circle Sym', 'Gnomonic'),	('Fuller', 'Gnomonic'),	('Bisect 2', 'Gnomonic'),	('Areal', 'Gnomonic'),	('Trilinear', 'Gnomonic'),	('Bisect', 'Gnomonic'),	
('Snyder Sym', 'Conformal'),	('Small Circle Sym', 'Conformal'),	('Fuller', 'Conformal'),	('Bisect 2', 'Conformal'),	('Areal', 'Conformal'),	('Trilinear', 'Conformal'),	('Bisect', 'Conformal'),	('Gnomonic', 'Conformal'),
        ]

ts = {}
sigmas = {}
for name1, name2 in pairs:
    degpts1 = degptss[name1]
    degpts2 = degptss[name2]
    def objective(t):
        degpts = t*degpts1 + (1-t)*degpts2
        omega, scale = mapstuff.omegascale(adegpts, degpts,
                                           geod, spacing=spacing)
        #scale[scale < 0] = np.nan
        scale[~insidemask] = np.nan
        return np.nanmax(scale)/np.nanmin(scale)
    result = minimize_scalar(objective, [1, 2], tol=1E-12)
    newname = name1 + ' + ' + name2
    ts[newname] = result.x
    sigmas[newname] = result.fun
    print(result.message)
    
datflation = pd.DataFrame([pd.Series(ts), pd.Series(sigmas)]).T
datflation.columns = ['t', 'flation ratio']
datflation.sort_values('flation ratio')
#%% interpolated angle
tsa = {}
omegas = {}
for name1, name2 in pairs:
    degpts1 = degptss[name1]
    degpts2 = degptss[name2]
    def objective(t):
        degpts = t*degpts1 + (1-t)*degpts2
        omega, scale = mapstuff.omegascale(adegpts, degpts,
                                           geod, spacing=spacing)
        omega[~insidemask] = np.nan
        #weight so the average comes out right
        if np.any(omega < 0):
            return 1E6
        avgm = np.nansum(omega*wght)/np.nansum(wght)
        return avgm
    result = minimize_scalar(objective, [1, 2], tol=1E-12)
    newname = name1 + ' + ' + name2
    tsa[newname] = result.x
    omegas[newname] = result.fun
    print(result.message)

datangle = pd.DataFrame([pd.Series(tsa), pd.Series(omegas)]).T
datangle.columns = ['t', 'avg omega']
datangle.sort_values('avg omega')

#%% interpolated cusp
cuspts = {}
cuspity = {}
for name1, name2 in pairs:
    e2e1 = e2es[name1][..., 2]
    e2e2 = e2es[name2][..., 2]
    #degpts1 = degptss[name1]
    #degpts2 = degptss[name2]
    def objective(t):
        e2e_t = t*e2e1 + (1-t)*e2e2
        cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                                 e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
        cpdiff = np.where(cp < 0, cp + 360, cp) - expected[2]
        return abs(cpdiff).max(axis=0)

    result = minimize_scalar(objective, [1, 2], tol=1E-12)
    newname = name1 + ' + ' + name2
    cuspts[newname] = result.x
    cuspity[newname] = result.fun
    print(result.message)    

datcusp = pd.DataFrame([pd.Series(cuspts), pd.Series(cuspity)]).T
datcusp.columns = ['t', 'cusp']
datcusp.sort_values('cusp')
#%% multivar optimization
namelist = [ 'Snyder Sym', 
             'Small Circle Sym', 
             'Fuller', 
             #'Bisect 2',  
             'Areal', 
             'Trilinear', 
             #'Bisect',  
             'Gnomonic', ]
             #'Conformal']
degptsx = np.stack([degptss[name] for name in namelist])
e2ex = np.stack([e2es[name][..., 2] for name in namelist])

# 1.030959016704561 w/o conformal
#initial_f = np.array([ 2.17 , -0.854,  0.052, -0.287,  0.523, -0.64 , -0.02, 0.056 ])
# 1.0310531146332595 with conformal
#initial_f = np.array([ 2.149, -0.866,  0.047, -0.278,  0.626, -0.654, -0.046,  0.014])
# 1.0344186013545849 with conformal, without bisect
#initial_f = np.array([ 2.015e+00, -1.021e+00,  1.000e-03, -4.500e-02,  4.500e-02, -1.000e-03])
# 1.0345459290249237 w/o conformal or bisect
initial_f = np.array([ 2.013, -1.021,  0, 0,  0])
def objective_f(t):
    params = np.concatenate([t, [1 - t.sum()]])
    degpts = np.tensordot(degptsx, params, axes = (0,0))
    omega, scale = mapstuff.omegascale(adegpts, degpts,
                                       geod, spacing=spacing)
    #scale[scale < 0] = np.nan
    scale[~insidemask] = np.nan
    return np.nanmax(scale)/np.nanmin(scale)

# 0.7628455111996661
#initial_a = np.array([-1.352, -3.217, 26.385, 16.492, -10.034, -15.094, -19.698])
# 0.8086459599880398 w/o conformal or bisect
initial_a = np.array([ -1.332,   6.773,  11.567, -14.16 ,  -7.365])
def objective_a(t):
    params = np.concatenate([t, [1 - t.sum()]])
    degpts = np.tensordot(degptsx, params, axes = (0,0))
    omega, scale = mapstuff.omegascale(adegpts, degpts,
                                       geod, spacing=spacing)
    omega[~insidemask] = np.nan
    #weight so the average comes out right
    if np.any(omega < 0):
        return 1E6
    avgm = np.nansum(omega*wght)/np.nansum(wght)
    return avgm

# 0.0030177224575993478
#initial_c = np.array([ 1.191, -0.272,  0.438,  0.675,  1.302, -0.8  , -0.843])
# 0.002925346218944469 w/o conformal or bisect
initial_c = np.array([ 1.377,  0.07 , -0.497,  0.159, -0.029])

def objective_c(t):
    params = np.concatenate([t, [1 - t.sum()]])
    e2e_t = np.tensordot(e2ex, params, axes = (0,0))
    cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                             e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
    cpdiff = np.where(cp < 0, cp + 360, cp) - expected[2]
    return abs(cpdiff).max(axis=0)

opt_f = initial_f
opt_a = initial_a
opt_c = initial_c

#%% flation
result = minimize(objective_f, opt_f, method = 'Nelder-Mead', options={
                  'xatol': 1E-14, 'fatol': 1E-14, 'maxiter':1E5,
                  'maxfev': 1E5})
print(result.fun, result.x, result.message)
opt_f = result.x
#%% angle
result = minimize(objective_a, opt_a, method = 'Nelder-Mead', options={
                  'xatol': 1E-14, 'fatol': 1E-14, 'maxiter':1E5,
                  'maxfev': 1E5})
print(result.fun, result.x, result.message)
opt_a = result.x
#%% cusp
result = minimize(objective_c, opt_c, method = 'Nelder-Mead', options={
                  'xatol': 1E-14, 'fatol': 1E-14, 'maxiter':1E6,
                  'maxfev': 1E6})
print(result.fun, result.x, result.message)
opt_c = result.x
#%% interpolated EA cusp
pairs = [('Snyder C', 'Snyder M'),					
('Snyder C', 'Snyder V'),	('Snyder M', 'Snyder V'),				
('Snyder C', 'Small Circle C'),	('Snyder M', 'Small Circle C'),	('Snyder V', 'Small Circle C'),			
('Snyder C', 'Small Circle M'),	('Snyder M', 'Small Circle M'),	('Snyder V', 'Small Circle M'),	('Small Circle C', 'Small Circle M'),		
('Snyder C', 'Small Circle V'),	('Snyder M', 'Small Circle V'),	('Snyder V', 'Small Circle V'),	('Small Circle C', 'Small Circle V'),	('Small Circle M', 'Small Circle V'),	]

cuspts = {}
cuspity = {}
for name1, name2 in pairs:
    e2e1 = e2es[name1]
    e2e2 = e2es[name2]
    #degpts1 = degptss[name1]
    #degpts2 = degptss[name2]
    def objective(t):
        e2e_t = t*e2e1 + (1-t)*e2e2
        cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                                 e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
        cpdiff = np.where(cp < 0, cp + 360, cp) - expected
        return abs(cpdiff).max()

    result = minimize_scalar(objective, [1, 2], tol=1E-12)
    newname = name1 + ' + ' + name2
    cuspts[newname] = result.x
    cuspity[newname] = result.fun
    print(result.message)    

datcusp = pd.DataFrame([pd.Series(cuspts), pd.Series(cuspity)]).T
datcusp.columns = ['t', 'cusp']
datcusp.sort_values('cusp')

#%% multivar optimization - EA cusps
namelist = ['Snyder C',
'Snyder M', 'Snyder V', 'Small Circle C', 'Small Circle M',
'Small Circle V' ]
degptsx = np.stack([degptss[name] for name in namelist])
e2ex = np.stack([e2es[name] for name in namelist])

#0.27506922787564747
initial_c = np.array([ 0.002, -0.032,  1.167,  0.284, -0.492])

def objective_c(t):
    params = np.concatenate([t, [1 - t.sum()]])
    e2e_t = np.tensordot(e2ex, params, axes = (0,0))
    cp =np.arctan2(e2e_t[0,:,1] - e2e_t[0,:,0],
                             e2e_t[1,:,1] - e2e_t[1,:,0])*180/np.pi
    cpdiff = np.where(cp < 0, cp + 360, cp) - expected
    return abs(cpdiff).max()

opt_c = initial_c
#%%
result = minimize(objective_c, opt_c, method = 'Nelder-Mead', options={
                  'xatol': 1E-14, 'fatol': 1E-14, 'maxiter':1E6,
                  'maxfev': 1E6})
print(result.fun, result.x, result.message)
opt_c = result.x
