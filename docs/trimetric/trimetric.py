#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:06:45 2019

@author: brsr
"""
import geopandas
import pandas as pd
import shapely
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import mapproj
import functools
#import os
# os.chdir('Code/mapproj')
np.seterr(divide='ignore')

#geod = pyproj.Geod(ellps='WGS84')
r = 6371
geod = pyproj.Geod(a=r)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
grat = mapproj.graticule()


def circgen(lon, lat, r, cross=False):
    az = 0
    circs = []
    for x, y, ri in zip(lon, lat, r):
        circ = []
        for az in range(361):
            out = geod.fwd(x, y, az, ri)
            circ.append(out[:2])
        circs.append(LineString(circ))
        if cross:
            for ax in [0, 180], [90, 270]:
                p1 = geod.fwd(x, y, ax[0], ri)[:2]
                p2 = geod.fwd(x, y, ax[1], ri)[:2]
                crox = geod.npts(p1[0], p1[1], p2[0], p2[1], 181)
                circs.append(LineString(crox))
    return geopandas.GeoSeries(circs)


cyclic = [0, 1, 2, 0]
pnames = ['Chamberlin', 'Matrix']
adegpts = np.array(np.meshgrid(np.linspace(-179.5, 179.5, 360),
                               np.linspace(-89.5, 89.5, 180)))
degpts = mapproj.arraytoptseries(adegpts)
degpts.crs = world.crs
wght = np.cos(adegpts[1]*np.pi/180)

control_points = {
    'Canada Atlas': geopandas.GeoSeries([Point(-98-13/60, 61+39/60), Point(-135, 40), Point(-55, 40)]),
    'Canada Wall Map': geopandas.GeoSeries([Point(-150, 60), Point(-97.5, 50), Point(-45, 60)]),
    'NW South America': geopandas.GeoSeries([Point(-69, -25), Point(-55, 10), Point(-85, 10)]),
    'Australia': geopandas.GeoSeries([Point(134, -8), Point(110, -32), Point(158, -32)]),
    'S South America': geopandas.GeoSeries([Point(-43, -18), Point(-72, -18), Point(-72, -56)]),
    'E South America': geopandas.GeoSeries([Point(-63-33/60, 8+8/60),
                                  Point(-58-33/60, -34-35/60), Point(-35-13/60, -5-47/60)]),
    'Europe Wall Map': geopandas.GeoSeries([Point(15, 72), Point(-8, 33), Point(38, 33)]),
    #'Strebe Africa': geopandas.GeoSeries([Point(0,22), Point(22.5, -22), Point(45, 22)]),
    'South America Wall Map': geopandas.GeoSeries([Point(-80, 9), Point(-71, -53), Point(-35, -6)]),
    'North America Wall Map': geopandas.GeoSeries([Point(-150, 55), Point(-92.5, 10), Point(-35, 55)]),
    'Africa Wall Map': geopandas.GeoSeries([Point(-19-3/60, 24+25/60), Point(20, -35), Point(59+3/60, 24+25/60)]),
    #'Hemisphere': geopandas.GeoSeries([Point(-180,0), Point(0, -60), Point(0, 60)]),
}

focus = {
    'Canada Atlas': world.index[world.name == 'Canada'],
    'Canada Wall Map': world.index[world.name == 'Canada'],
    'NW South America': world.index[world.name.isin(['Peru', 'Ecuador',
                                                     'Colombia', 'Suriname', 'Venezuela', 'Guyana'])],
    'Australia': world.index[world.name == 'Australia'],
    'S South America': world.index[world.name.isin(['Bolivia', 'Chile',
                                                    'Argentina', 'Uruguay', 'Paraguay'])],
    'E South America': world.index[world.name == 'Brazil'],
    # exclude France because of French Guinea and
    # Russia because half of it is in Asia
    'Europe Wall Map': world.index[(world.continent == 'Europe') &
                                   (world.name != 'France') &
                                   (world.name != 'Russia')],
    'Strebe Africa': world.index[world.continent == 'Africa'],
    'South America Wall Map': world.index[world.continent == 'South America'],
    'North America Wall Map': world.index[world.continent == 'North America'],
    'Africa Wall Map': world.index[world.continent == 'Africa'],
}

exclude = {
    'Canada Atlas': world.index[world.continent.isin(['Antarctica', 'Oceania'])],
    'Canada Wall Map': world.index[world.continent.isin(['Antarctica', 'Oceania'])],
    'NW South America': world.index[world.continent.isin(['Asia', 'Oceania'])],
    'Australia': world.index[world.continent.isin(['North America',
                                                   'South America',
                                                   'Africa', 'Europe'])],
    'S South America': world.index[world.continent.isin(['Asia'])],
    'E South America': world.index[world.continent.isin(['Asia', 'Oceania'])],
    'Europe Wall Map': world.index[world.continent.isin(['Antarctica', 'Oceania'])],
    'Strebe Africa': world.index[world.continent.isin(['North America'])],
    'South America Wall Map': world.index[world.continent.isin(['Asia', 'Oceania'])],
    'North America Wall Map': world.index[world.continent.isin(['Antarctica', 'Oceania'])],
    'Africa Wall Map': world.index[world.continent.isin(['North America'])],
    'Hemisphere': world.index[~world.continent.isin(['North America',
                                                     'South America'])],
}

extents = {'South America Wall Map': ([-3500, 4500], [-4000, 4000])}
scalelevels = {'South America Wall Map': np.linspace(0.98, 1.12, 8)}
omegalevels = {'South America Wall Map': np.linspace(0, 10, 6)}
dlevels = {'South America Wall Map': np.linspace(0, 100, 3)}

for controlpts in control_points.values():
    controlpts.crs = world.crs

pdindex = pd.MultiIndex.from_product(
    [control_points.keys(), pnames], names=['ctrlpts', 'proj'])
# %% make sure the control points are set up right: omit
for name in control_points:
    fig, ax = plt.subplots(figsize=(10, 5))
    world.plot(ax=ax, color='#B4B4B4')
    grat.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
    ctrlpts = control_points[name]
    ctrlpts.plot(ax=ax, color='green', marker='x')
    actrlpts = mapproj.ptseriestoarray(ctrlpts)
    antipodes = np.array([actrlpts[0] - 180, -actrlpts[1]])
    antipodes[0] = antipodes[0] + np.where(antipodes[0] < -180, 360, 0)
    ax.scatter(antipodes[0], antipodes[1])
    ax.set_title(name)
    ax.axis('equal')

# %%
cptable = pd.DataFrame(dtype=float,
                       columns=['pt1_lon', 'pt1_lat',
                                'pt2_lon', 'pt2_lat',
                                'pt3_lon', 'pt3_lat',
                                'len23', 'len31', 'len12', 'area'])
cyclic2 = [1,2,0,1]
pdtable = pd.DataFrame(index=pdindex, dtype=float,
                       columns=['avgomega', 'maxomega',
                                'avgld', 'maxld', 'minscale', 'maxscale', 'scalerat'])

for name, controlpts in control_points.items():
    print(name)
    actrlpts = mapproj.ptseriestoarray(controlpts)
    # stats about the control triangle
    assert np.linalg.det(mapproj.UnitVector.transform(
        actrlpts[0], actrlpts[1])) > 0
    cptable.loc[name] = np.concatenate([actrlpts.T.flatten(),
                                        geod.line_lengths(
                                            actrlpts[0, cyclic2], actrlpts[1, cyclic2]),
                                        [geod.polygon_area_perimeter(actrlpts[0], actrlpts[1])[0]]])

    # determine index for interior of triangle
    ar = mapproj.Areal(actrlpts, geod)
    adegpts_ar = ar.transform_v(adegpts)
    zone_ar = np.signbit(adegpts_ar).sum(axis=0)
    index_ar = zone_ar == 0
    #index_ar[0] = False
    #index_ar[-1] = False
    wght_m = wght.copy()
    wght_m[~index_ar] = np.nan
    # initialize projections
    chamstring = {'proj': 'chamb',
                  'lon_1': actrlpts[0, 0],
                  'lon_2': actrlpts[0, 2],
                  'lon_3': actrlpts[0, 1],
                  'lat_1': actrlpts[1, 0],
                  'lat_2': actrlpts[1, 2],
                  'lat_3': actrlpts[1, 1],
                  'R': geod.a}
    ct = mapproj.ChambTrimetric(actrlpts, geod)
    lt = mapproj.LinearTrimetric(actrlpts, geod)
    tgtpts = lt.tgtpts
    # various quantities to map
    gd = mapproj.geodesics(actrlpts[0], actrlpts[1], geod, n=100)
    gd.crs = world.crs
    antipodes = np.array([actrlpts[0] - 180, -actrlpts[1]])
    antipodes[0] = antipodes[0] + np.where(antipodes[0] < -180, 360, 0)
    center = lt.center
    gratrange = np.array([-90, 90]) + center[0]
    # to avoid the antipodal part
    grat2 = mapproj.graticule(lonrange=gratrange)
    cpts = np.array(np.meshgrid(np.linspace(-60, 60, 9) + center[0],
                                np.linspace(-60, 60, 9) +
                                np.clip(center[1], -25, 25))).reshape(2, -1)
    r = np.ones(cpts.shape[1])*500
    tissot = circgen(cpts[0], cpts[1], r, cross=True)
    tissot.crs = world.crs
    # perform the projections
    adegpts_lt = lt.transform_v(adegpts)
    controlpts_lt = mapproj.transeach(lt.transform, controlpts)
    world_lt = mapproj.transeach(lt.transform, world.geometry)
    grat_lt = mapproj.transeach(lt.transform, grat)
    grat2_lt = mapproj.transeach(lt.transform, grat2)
    gd_lt = mapproj.transeach(lt.transform, gd)
    tissot_lt = mapproj.transeach(lt.transform, tissot)

    #antigd_lt = mapproj.transeach(lt.transform, antigd)
    adegpts_rt = lt.invtransform_v(adegpts_lt).reshape((adegpts.shape))

    adegpts_ct = mapproj.ptseriestoarray(
        degpts.to_crs(chamstring)).reshape(adegpts_lt.shape)
    # ct.transform_v(adegpts)
    # mapproj.transeach(ct.transform, controlpts)
    controlpts_ct = controlpts.to_crs(chamstring)
    # mapproj.transeach(ct.transform, world.geometry)
    world_ct = world.to_crs(chamstring)
    grat_ct = grat.to_crs(chamstring)  # mapproj.transeach(ct.transform, grat)
    # mapproj.transeach(ct.transform, grat2)
    grat2_ct = grat2.to_crs(chamstring)
    gd_ct = gd.to_crs(chamstring)  # mapproj.transeach(ct.transform, gd)
    # mapproj.transeach(ct.transform, tissot)
    tissot_ct = tissot.to_crs(chamstring)

    # affine transform ct to line up with lt
    #ltz = mapproj.ptseriestoarray(controlpts_lt).T.copy().view(dtype=complex)
    #ctz = mapproj.ptseriestoarray(controlpts_ct).T.copy().view(dtype=complex)
    #a = ((ctz[0]-ctz[1])/(ltz[0] - ltz[1]))[0]
    #b = (ctz[0] - a*ltz[0])
    #amat = np.array([[a.real, a.imag],[-a.imag, a.real]])
    #bvec = b.view(dtype=float)
    #affine = [a.real, a.imag, -a.imag, a.real, b.real, b.imag]
    uno = np.ones((1, 3))
    ltz = np.concatenate([mapproj.ptseriestoarray(controlpts_lt), uno])
    ctz = np.concatenate([mapproj.ptseriestoarray(controlpts_ct), uno])
    # want matrix such that ltz = M @ ctz
    #M = ltz @ inv(ctz)
    # this isn't necessarily numerically stable but oh well
    matrix = ltz @ np.linalg.inv(ctz)
    affine = [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
              matrix[0, 2], matrix[1, 2]]
    adegpts_ct = (np.tensordot(matrix[:2, :2], adegpts_ct, axes=(1, 0))
                  + matrix[:2, 2][:, np.newaxis, np.newaxis])
    controlpts_ct = controlpts_ct.affine_transform(affine)
    world_ct = world_ct.affine_transform(affine)
    grat_ct = grat_ct.affine_transform(affine)
    grat2_ct = grat2_ct.affine_transform(affine)
    gd_ct = gd_ct.affine_transform(affine)
    tissot_ct = tissot_ct.affine_transform(affine)

    adegpts_rc = lt.invtransform_v(adegpts_ct).reshape(adegpts.shape)

    # distortion calculations
    omega_lt, scale_lt = mapproj.omegascale(adegpts, adegpts_lt, geod)
    omega_ct, scale_ct = mapproj.omegascale(adegpts, adegpts_ct, geod)

    omegas = [omega_ct, omega_lt]
    scales = [scale_ct, scale_lt]

    unprojlengths = []
    for i in range(3):
        f, b, l = geod.inv(adegpts[0], adegpts[1],
                           actrlpts[0, i]*np.ones(adegpts.shape[1:]),
                           actrlpts[1, i]*np.ones(adegpts.shape[1:]))
        unprojlengths.append(l)
    unprojlengths = np.array(unprojlengths)

    lengthdistorts = []
    for dp in [adegpts_ct, adegpts_lt]:
        x = dp[:, np.newaxis] - tgtpts[..., np.newaxis, np.newaxis]
        y = mapproj.sqrt(np.sum(x**2, axis=0))
        result = np.nansum(abs(y - unprojlengths), axis=0)
        lengthdistorts.append(result)

    for omega, scale, ld, pname in zip(omegas, scales, lengthdistorts, pnames):
        avgomega = np.nansum(omega*wght_m)/np.nansum(wght_m)
        maxomega = np.nanmax(omega[index_ar])
        avgld = np.nansum(ld*wght_m)/np.nansum(wght_m)
        maxld = np.nanmax(ld[index_ar])
        avgscale = np.nansum(scale*wght_m)/np.nansum(wght_m)
        minscale = np.nanmin(scale[index_ar])
        maxscale = np.nanmax(scale[index_ar])
        scalerat = maxscale/minscale - 1
        #print(avgscale, minscale, maxscale)
        pdtable.loc[name, pname] = [avgomega, maxomega, avgld, maxld,
                                    minscale, maxscale, scalerat*100]
    # %
    excl = exclude[name]
    world_ctv = world_ct.drop(excl)
    world_ltv = world_lt.drop(excl)
    # %#figure: comparison of projections (omit, use the next one)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax = axes[0]
    world_ctv.plot(ax=ax, color='k')
    grat_ct.plot(ax=ax, color='lightgrey', linewidth=1)
    controlpts_ct.plot(ax=ax, color='green', marker='x')
    gd_ct.plot(ax=ax, color='green', linestyle=':')
    ax.set_title('Chamberlin')
    ax = axes[1]
    world_ltv.plot(ax=ax, color='k')
    grat_lt.plot(ax=ax, color='lightgrey', linewidth=1)
    controlpts_lt.plot(ax=ax, color='green', marker='x')
    gd_lt.plot(ax=ax, color='green', linestyle=':')
    ax.set_title('Matrix')
    fig.savefig(name + '_whole.png')
    # bounds for plots
    try:
        xbounds, ybounds = extents[name]
    except KeyError:
        bd = pd.concat([gd_ct.bounds, gd_lt.bounds])
        try:
            bd = pd.concat([bd,
                            world_ct.loc[focus[name]].bounds,
                            world_lt.loc[focus[name]].bounds])
        except KeyError:
            pass
        bd[~np.isfinite(bd)] = np.nan
        xbounds = [bd.minx.min(), bd.maxx.max()]
        xbounds[0] = xbounds[0]*1.1 if xbounds[0] < 0 else xbounds[0]*0.9
        xbounds[1] = xbounds[1]*1.1 if xbounds[1] > 0 else xbounds[1]*0.9
        ybounds = [bd.miny.min(), bd.maxy.max()]
        ybounds[0] = ybounds[0]*1.1 if ybounds[0] < 0 else ybounds[0]*0.9
        ybounds[1] = ybounds[1]*1.1 if ybounds[1] > 0 else ybounds[1]*0.9
        xsize = xbounds[1] - xbounds[0]
        ysize = ybounds[1] - ybounds[0]
        if xsize > ysize:
            ymean = tgtpts[1].mean()
            ybounds = ymean - xsize/2, ymean + xsize/2,
    # %#figure: zoom comparison of projections
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    ax = axes[0]
    world_ctv.plot(ax=ax, color='k')
    grat2_ct.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
    gd_ct.plot(ax=ax, color='green', linestyle=':')
    controlpts_ct.plot(ax=ax, color='green', marker='x')
    ax.set_title('Chamberlin')
    ax = axes[1]
    world_ltv.plot(ax=ax, color='k')
    grat2_lt.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
    gd_lt.plot(ax=ax, color='green', linestyle=':')
    controlpts_lt.plot(ax=ax, color='green', marker='x')
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    ax.set_title('Matrix')
    fig.savefig(name + '_zoom.eps')
    # %#figure: tissot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    axes[0].set_title('Chamberlin')
    axes[1].set_title('Matrix')
    for gdx, worldx, grat2x, tissots, ax in zip(
        [gd_ct, gd_lt],
        [world_ctv, world_ltv],
        [grat2_ct, grat2_lt],
            [tissot_ct, tissot_lt], axes):
        worldx.plot(ax=ax, color='#B4B4B4')
        grat2x.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
        tissots.plot(ax=ax, color='k')
        ax.scatter(tgtpts[0], tgtpts[1], color='g', marker='x')
        ax.axis('equal')
        gdx.plot(ax=ax, color='g', linestyle=':')
        ax.tick_params(axis='y', labelrotation=90)
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    plt.draw()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), va='center')
    fig.savefig(name + '_tissot.eps')
    # %#figure: scale
    try:
        levels = scalelevels[name]
    except KeyError:
        px = pdtable.loc[name]
        pmax = px.maxscale.max()
        pmin = np.clip(px.minscale.min(), 0, pmax)
        levels = np.linspace(pmin, pmax, 6)

    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.1]},
                             figsize=(10, 4.5))
    axes[0].set_title('Chamberlin')
    axes[1].set_title('Matrix')
    axes[0].get_shared_y_axes().join(axes[0], axes[1])
    axes[1].set_yticklabels([])
    rindex = (scales[0] > 0) & (scales[1] > 0)
    for scale, dp, gdx, worldx, grat2x, ax in zip(scales,
                                                  [adegpts_ct, adegpts_lt],
                                                  [gd_ct, gd_lt],
                                                  [world_ctv, world_ltv],
                                                  [grat2_ct, grat2_lt], axes):
        worldx.plot(ax=ax, color='#B4B4B4')
        grat2x.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
        cs = ax.contour(dp[0], dp[1],
                        np.where(rindex, scale, np.nan),
                        colors='k', levels=levels)
        ax.clabel(cs, fmt='%1.2f', inline_spacing=-8)
        thing = scale.copy()
        thing[~index_ar] = np.nan
        ah = np.unravel_index(np.nanargmax(thing), thing.shape)
        al = np.unravel_index(np.nanargmin(thing), thing.shape)
        ax.scatter(dp[0][al], dp[1][al], color='k', marker='+')
        ax.scatter(tgtpts[0], tgtpts[1], color='g', marker='x')
        ax.axis('equal')
        gdx.plot(ax=ax, color='g', linestyle=':')
        ax.tick_params(axis='y', labelrotation=90)
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    plt.draw()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), va='center')
    fig.colorbar(cs, axes[2])
    fig.savefig(name + '_scale.eps')
    # %#figure: omega
    try:
        levels = omegalevels[name]
    except KeyError:
        levels = np.linspace(0, np.floor(px.maxomega.max()), 6)
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.1]},
                             figsize=(10, 4.5))
    axes[0].set_title('Chamberlin')
    axes[1].set_title('Matrix')
    axes[0].get_shared_y_axes().join(axes[0], axes[1])
    axes[1].set_yticklabels([])
    param = omegas[0] - omegas[1]
    for omega, dp, gdx, worldx, grat2x, ax in zip(omegas,
                                                  [adegpts_ct, adegpts_lt],
                                                  [gd_ct, gd_lt],
                                                  [world_ctv, world_ltv],
                                                  [grat2_ct, grat2_lt], axes):
        worldx.plot(ax=ax, color='#B4B4B4')
        grat2x.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
        cs = ax.contour(dp[0], dp[1],
                        np.where(rindex, omega, np.nan), colors='k',
                        levels=levels)
        ax.clabel(cs, fmt='%1.0f', inline_spacing=0)
        thing = omega.copy()
        thing[~index_ar] = np.nan
        ah = np.unravel_index(np.nanargmax(thing), thing.shape)
        al = np.unravel_index(np.nanargmin(thing), thing.shape)
        ax.scatter(dp[0][al], dp[1][al], color='k', marker='+')
        ax.scatter(tgtpts[0], tgtpts[1], color='g', marker='x')
        ax.axis('equal')
        gdx.plot(ax=ax, color='g', linestyle=':')
        ax.tick_params(axis='y', labelrotation=90)
        ax.contour(dp[0], dp[1],
                   np.where(rindex, param, np.nan),
                   levels=[0], colors='b', linestyles='--')
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    plt.draw()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), va='center')
    fig.colorbar(cs, axes[2])
    fig.savefig(name + '_omega.eps')
    # %#figure: deviation in distance
    try:
        levels = dlevels[name]
    except KeyError:
        levels = np.linspace(0, np.ceil(px.maxld.max()), 6)
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.1]},
                             figsize=(10, 4.5))
    axes[0].set_title('Chamberlin')
    axes[1].set_title('Matrix')
    axes[0].get_shared_y_axes().join(axes[0], axes[1])
    axes[1].set_yticklabels([])
    param = lengthdistorts[0] - lengthdistorts[1]
    for ld, dp, gdx, worldx, grat2x, ax in zip(lengthdistorts,
                                               [adegpts_ct, adegpts_lt],
                                               [gd_ct, gd_lt],
                                               [world_ctv, world_ltv],
                                               [grat2_ct, grat2_lt], axes):
        worldx.plot(ax=ax, color='#B4B4B4')
        grat2x.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
        cs = ax.contour(dp[0], dp[1],
                        np.where(rindex, ld, np.nan),
                        colors='k', levels=levels)
        ax.clabel(cs, fmt='%1.0f', inline_spacing=-2)
        thing = ld.copy()
        thing[~index_ar] = np.nan
        ah = np.unravel_index(np.nanargmax(thing), thing.shape)
        #al = np.unravel_index(np.nanargmin(thing), thing.shape)
        ax.scatter(dp[0][ah], dp[1][ah], color='k', marker='+')
        ax.scatter(tgtpts[0], tgtpts[1], color='g', marker='x')
        gdx.plot(ax=ax, color='g', linestyle=':')
        ax.axis('equal')
        ax.tick_params(axis='y', labelrotation=90)
        ax.contour(dp[0], dp[1],
                   np.where(rindex, param, np.nan),
                   levels=[0], colors='b', linestyles='--')
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    plt.draw()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), va='center')
    fig.colorbar(cs, axes[2])
    fig.savefig(name + '_distance.eps')
# %%
#pdtable.loc['Hemisphere', 'scalerat'] = np.inf
#pdtable.loc['Hemisphere', 'maxomega'] = 180
pdtable.to_csv('cham_matr_stats.csv', index_label='name')
cptable.to_csv('control_triangles.csv', index_label='name')
pdtablenoh = pdtable#.drop('Hemisphere')
area = cptable.area#.drop('Hemisphere')
#sl = cptable[['len12','len23','len31']].drop('Hemisphere')
#aspect = sl.max(axis=1) - sl.min(axis=1)
cham_maxo = pdtablenoh.xs('Chamberlin', level=1)['maxomega']
matrix_maxo = pdtablenoh.xs('Matrix', level=1)['maxomega']
cham_avgo = pdtablenoh.xs('Chamberlin', level=1)['avgomega']
matrix_avgo = pdtablenoh.xs('Matrix', level=1)['avgomega']
cham_maxd = pdtablenoh.xs('Chamberlin', level=1)['maxld']
matrix_maxd = pdtablenoh.xs('Matrix', level=1)['maxld']
cham_avgd = pdtablenoh.xs('Chamberlin', level=1)['avgld']
matrix_avgd = pdtablenoh.xs('Matrix', level=1)['avgld']
cham_sr = pdtablenoh.xs('Chamberlin', level=1)['scalerat']
matrix_sr = pdtablenoh.xs('Matrix', level=1)['scalerat']
labels = area.index
ticks = np.arange(len(labels))
#%%
cptablefmt = cptable.copy()
lens = ['len23', 'len31', 'len12']
cptablefmt[lens] = cptable[lens].round()
cptablefmt.area = (cptable.area/1E6).round(2)

def decdeg2dm(dd, suffix=['W','E']):
    if dd == 0:
        return '0\\degree'
    elif dd >= 0:
        suf = suffix[1]
    else:
        suf = suffix[0]
    dd = abs(dd)
    degrees,minutes = divmod(dd*60,60)
    dstring = "{}\\degree".format(degrees)
    mstring = "{}\'".format(minutes) if minutes != 0 else ''
    return dstring + mstring + suf
lons = ['pt1_lon', 'pt2_lon', 'pt3_lon']
lats = ['pt1_lat', 'pt2_lat', 'pt3_lat']
cptablefmt[lons] = np.vectorize(decdeg2dm)(cptable[lons].values)
dd2 = functools.partial(decdeg2dm, suffix=['S','N'])
cptablefmt[lats] = np.vectorize(dd2)(cptable[lats].values)
cptablefmt.to_csv('control_triangles.csv', index_label='name')
# %% scatter plots of the previous: omega
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(b=True, which='major', color='#666666', linestyle='-', axis='x')
ax.scatter(cham_maxo, ticks, edgecolor='k', zorder=6,
           color='lightblue', label='Chamberlin max')
ax.scatter(cham_avgo, ticks, edgecolor='k', zorder=6,
           color='dodgerblue', label='Chamberlin average')
ax.scatter(matrix_maxo, ticks, edgecolor='k',  marker='s', zorder=5,
           color='orange', label='Matrix max')
ax.scatter(matrix_avgo, ticks, edgecolor='k',  marker='s', zorder=5,
           color='orangered', label='Matrix average')
ax.set_xlabel('$\omega$, degrees')
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.legend(loc='best')
fig.subplots_adjust(left=0.3)
fig.savefig('omegaplot.eps')
# %%D
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(b=True, which='major', color='#666666', linestyle='-', axis='x')
#ax.scatter(cham_maxd, ticks, edgecolor='k', zorder=6,
#           color='lightblue', label='Chamberlin max')
ax.scatter(matrix_maxd, ticks, edgecolor='k',  marker='s', zorder=5,
           color='lightgrey', label='Both max')
ax.scatter(cham_avgd, ticks, edgecolor='k', zorder=6,
           color='dodgerblue', label='Chamberlin average')
ax.scatter(matrix_avgd, ticks, edgecolor='k',  marker='s', zorder=5,
           color='orangered', label='Matrix average')
ax.set_xlabel('$D$, km')
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.legend(loc='best')
fig.subplots_adjust(left=0.3)
fig.savefig('distanceplot.eps')
# %% scale
fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(b=True, which='major', color='#666666', linestyle='-', axis='x')
ax.scatter(cham_sr, ticks, edgecolor='k', color='dodgerblue',
           label='Chamberlin', zorder=6)
ax.scatter(matrix_sr, ticks, edgecolor='k', color='orange', marker='s',
           label='Matrix', zorder=5)
ax.set_xlabel('scale ratio, %')
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.legend(loc='best')
ax.set_xlim(left=0)
fig.subplots_adjust(left=0.3)
fig.savefig('scaleplot.eps')
# %% ~~~~~~~~~~~~~~~~~~~~~
testpt = [-70, -5]
f, b, r = geod.inv(testpt[0]*np.ones(actrlpts.shape[1]),
                   testpt[1]*np.ones(actrlpts.shape[1]),
                   actrlpts[0], actrlpts[1])

az = 0
circs = circgen(actrlpts[0], actrlpts[1], r)
stestpt = geopandas.GeoSeries(Point(testpt))
stestpt.crs = world.crs
sctpt = stestpt.to_crs(chamstring).affine_transform(affine)
ctpt = np.array(sctpt[0].xy)
#ctpt = ct.transform(testpt[0], testpt[1])
ltpt = lt.transform(testpt[0], testpt[1])

actrlpts_o = mapproj.UnitVector.transform_v(actrlpts)
# %%#figure: show construction
fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
ax = axes[0]
#world.plot(ax=ax, color='k')
grat.plot(ax=ax, color='lightgrey', linewidth=1, alpha=0.5)
controlpts.plot(ax=ax, color='green', marker='x')
gd.plot(ax=ax, color='green', linestyle=':')
circs.plot(ax=ax, color='b')
ax.scatter(*testpt, color='b')
#antigd.plot(ax=ax, color='yellow')
ax.set_aspect('equal', 'box')
ax.set_xlim(-105, -15)
ax.set_ylim(-60, 30)
xticks = ['$90\\degree$W','$60\\degree$W','$30\\degree$W',]
yticks = ['$60\\degree$S', '$30\\degree$S', '$0\\degree$', '$30\\degree$N']
ax.set_xticks(np.linspace(-90, -30, 3))
ax.set_xticklabels(xticks)
ax.set_yticks(np.linspace(-60, 30, 4))
ax.set_yticklabels(yticks)
ax.annotate("$v_1$", actrlpts[:, 0], xytext=(0, 5), ha='right',
            textcoords="offset points")
ax.annotate("$v_2$", actrlpts[:, 1], xytext=(5, -5), ha='left',
            textcoords="offset points")
ax.annotate("$v_3$", actrlpts[:, 2], xytext=(5, -5), ha='left',
            textcoords="offset points")
ax.annotate("$v$", testpt, xytext=(8, -10), ha='center',
            textcoords="offset points")
ax.set_title('a')
ax = axes[1]
#world_lt.plot(ax=ax, color='k')
#grat2_lt.plot(ax=ax, color='lightgrey', linewidth=1)
#gd_lt.plot(ax=ax, linestyle=':', color='green')
ax.plot(tgtpts[0, cyclic], tgtpts[1, cyclic], color='green',
        marker='x', linestyle=':')
for xi, yi, ri in zip(tgtpts[0], tgtpts[1], r):
    c = plt.Circle((xi, yi), ri, color='b', fill=False)
    ax.add_artist(c)

ri2 = np.roll(r, 1)**2
rj2 = np.roll(r, -1)**2
zi = np.roll(tgtpts, 1, axis=-1)
zj = np.roll(tgtpts, -1, axis=-1)
# should work as long as xi != xj
y = np.array([-5E6, 5E6])[..., np.newaxis]
x = ((ri2 - rj2)/2 + y*(zi[1]-zj[1]))/(zj[0]-zi[0])
ax.plot(x, y, color='r', linestyle='--')
ax.scatter(ctpt[0], ctpt[1], color='b')
ax.scatter(ltpt[0], ltpt[1], color='r')
ax.set_aspect('equal', 'box')
ax.set_xlim(-5E3, 5E3)
ax.set_ylim(-5E3, 5E3)
ax.annotate("$p_1$", tgtpts[:, 0], xytext=(0, 5), ha='right',
            textcoords="offset points")
ax.annotate("$p_2$", tgtpts[:, 1], xytext=(5, -5), ha='left',
            textcoords="offset points")
ax.annotate("$p_3$", tgtpts[:, 2], xytext=(5, -5), ha='left',
            textcoords="offset points")
ax.annotate("$p$", (ctpt[0], ctpt[1]), xytext=(0, 10),
            ha='left', textcoords="offset points")
ax.set_title('b')

# figure out how to inset this later?
ax = axes[2]
#gd_lt.plot(linestyle=':', ax=ax, color='green')
#ax.plot(tgtpts[0, index], tgtpts[1, index], color='green', marker='o')
for xi, yi, ri in zip(tgtpts[0], tgtpts[1], r):
    c = plt.Circle((xi, yi), ri, color='b', fill=False)
    ax.add_artist(c)
ax.scatter(ctpt[0], ctpt[1], color='b')
ax.scatter(ltpt[0], ltpt[1], color='r')
ax.plot(x, y, color='r', linestyle='--')
ax.set_aspect('equal', 'box')
#ax.set_xlim(ctpt[0] -500, ctpt[0]+500)
#ax.set_ylim(ctpt[1] -500, ctpt[1]+500)
ax.set_xlim(-750, -580)
ax.set_ylim(1550, 1700)
ax.annotate("$p_\ell$", (ltpt[0], ltpt[1]), xytext=(0, 8),
            ha='right', textcoords="offset points")
ax.annotate("$p_c$", (ctpt[0], ctpt[1]), xytext=(5, 0),
            ha='left', textcoords="offset points")
ax.set_title('c')
fig.savefig('construction.eps')
# %%#figure: analysis of variable h (omit)
cosrthmin = 1 / np.sqrt(lt.invperpmatrix.sum())
h0 = np.arccos(cosrthmin)**2

ppts = np.array([[0, 0],
                 [0, 9E3],
                 [0, 1.6E4],
                 [0, 1.8E4],
                 [6E3, 9E3],
                 [6E3, 0],
                 [1.2E4, 0]]).T
hp, nm = lt.nmforplot(ppts)
nm = nm - 1
fig, ax = plt.subplots(figsize=(10, 5))
pt = ax.plot(hp.T, nm.T)
ax.axhline(y=0, color='k')
ax.set_ylim([-1, 1])
ax.set_xlim([0, 8])
ax.set_xlabel('$h$')
ax.set_ylabel('$f(h)$')
ax.scatter(h0, 0)
# plt.legend(pt)
y = nm.min(axis=1)
x = hp[np.arange(7), nm.argmin(axis=1)]
for i, j, t in zip(x, y, [str(x) for x in ppts.T/1E3]):
    pos = np.array([np.clip(i, 0, 8), np.clip(j+0.05, -1, 1)])
    ax.annotate(t, pos, ha='center')

# %%#figure: antipodes (omit)
grat3 = mapproj.graticule(lonrange=[150, 255], latrange=[-60, 30])
fig, ax = plt.subplots(figsize=(10, 5), sharex=True, sharey=True)
for scale, linestyle in [(scale_ct, 'dashed'), (scale_lt, 'solid')]:
    cs = ax.contour(adegpts[0], adegpts[1], scale,
                    levels=0, linestyles=linestyle)
ax.scatter(antipodes[0], antipodes[1], color='k', marker='x')
world.plot(ax=ax, color='k')
grat3.plot(ax=ax, color='lightgrey', linewidth=1)
# %% test
def trigivenangles(angles, scale=np.pi/180):
    """Given angles, create the vertices of a triangle with those vertex
    angles. Only uses the first 2 angles. The last vertex is always 1, 0.
    >>> angles = np.array([45,90,45])
    >>> np.round(trigivenangles(angles), decimals=8)
    array([[-1.,  0.,  1.],
           [ 0., -1.,  0.]])
    """
    angles = angles * scale
    p0 = [np.cos(2*angles[1]), np.sin(2*angles[1])]
    p1 = [np.cos(2*angles[0]), np.sin(-2*angles[0])]
    p2 = [1, 0]
    return np.array([p0, p1, p2]).T

def anglesgivensides(sides, scale=180/np.pi):
    """Given side lengths of a triangle, determines the interior angle at each
    vertex, and the radius of the circumcircle.
    >>> sides=np.array( [3,4,5])
    >>> anglesgivensides(sides)
    """
    #might be more stable to use law of cotangents, but eh
    r = np.product(sides)/np.sqrt(
            2*np.sum(sides**2*np.roll(sides,1)**2)
            -np.sum(sides**4))
    s1 = sides
    s2 = np.roll(sides, -1)
    s3 = np.roll(sides, 1)
    cosangle = (s2**2 + s3**2 - s1**2)/ (2*s2*s3)
    angles = np.arccos(cosangle)
    return angles*scale, r

sides = np.array([8E6, 8E6, 8E6])

ang, r = anglesgivensides(sides)
px = r*trigivenangles(ang)

a = sides
b = np.roll(sides, 1)
c = np.roll(sides, -1)

s = sides.sum() / 2

A = np.sqrt(s*np.product(s-sides))
abc = sides.prod()
# R = sides.prod()/(4*A)

# cosphi = (b**2 + c**2 - a**2)/(2*b*c)
# # sin(2x)/2sin(x) = cos(x)
# # C sin(2 phi_1) = s_1 sin(2 phi_1)/2 sin(phi_1) = s_1 cos(phi_1)
# scp = a*cosphi

# # y = arccos(x), cos(2y)/2sin(y) = (2x^2 - 1)/(2*sqrt(1-x^2))
# # x = (c^2+b^2-a^2)/(2bc)

# # C cos(2 phi_i) = s_1 cos(2 phi_i)/2 sin(phi_i) =

# ccp = a *(b**2 * c**2 - 8*A**2 )/(4*b*c*A)

# px2 = np.array([[ccp[1],ccp[0],R],[scp[1],-scp[0],0]])


scpabcA = 2*A*a**2 * (b**2 + c**2 - a**2)
ccpabcA = a**2 * b**2 * c**2 - 8*a**2 * A**2 
RabcA = abc**2

px3 = np.array([[ccpabcA[1], ccpabcA[0], RabcA],
                [scpabcA[1],-scpabcA[0], 0]])/(4*A*abc)

