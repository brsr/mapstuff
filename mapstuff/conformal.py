# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:12:53 2023

@author: brsr
"""
import numpy as np
import pyproj
from scipy.optimize import minimize#, minimize_scalar, root_scalar
from scipy.special import hyp2f1, gamma#, ellipj, ellipk, ellipkinc

from .helper import sqrt, float2d_to_complex, complex_to_float2d, anglesgivensides
from .projections import CtrlPtsProjection, _unitsphgeod
from .transformations import Barycentric

def schwarz_fp(alpha, beta, gam):
    """Parameters of the Schwarz triangle map.
    Args:
        alpha, beta, gamma: Equal to pi times an angle of the triangle.
    Returns:
        s1: Value of the Schwarz triangle map at z=1.
        sinf: Value of the Schwarz triangle map at z=infinity.
        scale: Scale factor for spherical triangles. Will be zero or undefined
        if alpha + beta + gamma <= 1.
    """
    a = (1 - alpha - beta - gam)/2
    b = (1 - alpha + beta - gam)/2
    c = 1 - alpha
    palpha = np.pi*alpha
    pbeta = np.pi*beta
    pgam = np.pi*gam
    gfact = gamma(2-c)/(gamma(1-a)*gamma(c))
    s1 = gamma(c-a)*gamma(c-b)/gamma(1-b)*gfact
    sinf = np.exp(1j*palpha)*gamma(b)*gamma(c-a)*gfact/gamma(b-c+1)
    scale = sqrt(abs((np.cos(palpha+pbeta)+np.cos(pgam))/
                 (np.cos(palpha-pbeta)+np.cos(pgam))))
    return s1, sinf, scale

def c2c_mobius_finite(z,zi,wi):
    """Mobius transformation defined by mapping the points in zi to the points
    in wi."""
    ones = np.ones(zi.shape)
    a = np.linalg.det(np.stack([zi*wi,wi,ones]))
    b = np.linalg.det(np.stack([zi*wi,zi,wi]))
    c = np.linalg.det(np.stack([zi,wi,ones]))
    d = np.linalg.det(np.stack([zi*wi,zi,ones]))
    return (a*z+b)/(c*z+d)

def c2c_mobius_01inf(z, z0=0, z1=1, zinf=1j ):
    """Mobius transformation defined by mapping 3 points to 0, 1, infinity"""
    if ~np.isfinite(zinf):
        return (z-z0)/(z1-z0)
    elif ~np.isfinite(z1):
        return (z-z0)/(z-zinf)
    elif ~np.isfinite(z0):
        return (z1-zinf)/(z-zinf)
    else:
        return (z-z0)*(z1-zinf)/((z-zinf)*(z1-z0))

class ConformalTri(CtrlPtsProjection):
    nctrlpts = 3

    def __init__(self, ctrlpts, tgtpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod=geod)
        self.tgtpts = float2d_to_complex(tgtpts.T).squeeze()

        actrlpts = ctrlpts
        basei = 0
        basept = actrlpts[:, basei]
        crsst = {'proj': 'stere',
                 'lon_0': basept[0],
                 'lat_0': basept[1]}
        world_crs = {'init': 'epsg:4326'}
        stert = pyproj.transformer.Transformer.from_crs(world_crs,
                                                             crs_to=crsst)
        sterti = pyproj.transformer.Transformer.from_crs(crsst,
                                                         crs_to=world_crs)
        self.stert = stert
        self.sterti = sterti
        self.ctrl_s1, self.ctrl_sinf, self.ctrl_scale = schwarz_fp(*self.ctrl_angles/180)

        alpha, beta, gam = self.ctrl_angles/180
        self.a = (1 - alpha - beta - gam)/2
        self.b = (1 - alpha + beta - gam)/2
        self.c = 1 - alpha
        self.ap = (1 + alpha - beta - gam)/2#a - c + 1
        self.bp = (1 + alpha + beta - gam)/2#b - c + 1
        self.cp = 1 + alpha#2-c

        tgt_sides = abs(np.roll(self.tgtpts, 1) - np.roll(self.tgtpts, -1))
        tgt_angles = anglesgivensides(tgt_sides, scale=1)[0]
        alphat, betat, gamt = tgt_angles/np.pi
        self.apt = (1 + alphat - betat - gamt)/2
        self.bpt = (1 + alphat + betat - gamt)/2#
        self.cpt = 1 + alphat
        self.ct = 1 - alphat

        self.t1_s1, self.t1_sinf, _ = schwarz_fp(alphat, betat, gamt)

        self.pts_t = np.array(stert.transform(actrlpts[0], actrlpts[1]))
        self.pts_c = float2d_to_complex(self.pts_t.T.copy()).squeeze()
        #pts_r = pts_c / pts_c[1] * ctrl_s1
        self.bx = self.tgtpts[0]
        self.ax = (self.tgtpts[1] - self.tgtpts[0])/self.t1_s1

    def transform(self, lon, lat):
        lon + 0
        testpt_t = np.array(self.stert.transform(lon, lat))
        testpt_c = float2d_to_complex(testpt_t).squeeze()
        testpt_r = testpt_c / self.pts_c[1] * self.ctrl_s1

        a = self.a
        b = self.b
        c = self.c
        ap = self.ap
        bp = self.bp
        cp = self.cp

        def objective(t):
            z = t.view(dtype=complex)
            result = z**(1-c)*hyp2f1(ap,bp,cp,z)/hyp2f1(a,b,c,z)
            return abs(result - testpt_r)
        initial = c2c_mobius_01inf(testpt_r,
                                   z1=self.ctrl_s1, zinf=self.ctrl_sinf)
        res = minimize(objective, x0=[initial.real, initial.imag],
                       method='Nelder-Mead', options={'maxiter': 1E3})
        h = res.x.view(dtype=np.complex)

        ct = self.ct
        apt = self.apt
        bpt = self.bpt
        cpt = self.cpt
        testpt_t1 = h**(1-ct)*hyp2f1(apt,bpt,cpt,h)
        final = self.ax*testpt_t1 + self.bx
        return complex_to_float2d(final).T

    def invtransform(self, x, y):
        final = x + 1j*y
        testpt_t1i = (final - self.bx)/self.ax
        ct = self.ct
        apt = self.apt
        bpt = self.bpt
        cpt = self.cpt
        a = self.a
        b = self.b
        c = self.c
        ap = self.ap
        bp = self.bp
        cp = self.cp
        def objectivei(t):
            z = t.view(dtype=complex)
            result = z**(1-ct)*hyp2f1(apt,bpt,cpt,z)
            return abs(result - testpt_t1i)
        initiali = c2c_mobius_01inf(testpt_t1i,
                                    z1=self.t1_s1, zinf=self.t1_sinf)
        resi = minimize(objectivei, x0=[initiali.real, initiali.imag],
                       method='Nelder-Mead', options={'maxiter': 1E3})
        hi = resi.x.view(dtype=np.complex)

        testpt_ri = hi**(1-c)*hyp2f1(ap,bp,cp,hi)/hyp2f1(a,b,c,hi)
        testpt_ci = testpt_ri * self.pts_c[1]/self.ctrl_s1
        testpt_ti = complex_to_float2d(testpt_ci).T
        testpt_i = self.sterti.transform(*testpt_ti)
        return testpt_i

class ConformalTri3(CtrlPtsProjection):
    nctrlpts = 3

    def __init__(self, ctrlpts, tgtpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod=geod)
        self.tgtpts = float2d_to_complex(tgtpts.T).squeeze()
        subproj = []
        for i in range(3):
            rc = np.roll(ctrlpts, -i, axis=1)
            rt = np.roll(self.tgtpts, -i)
            subproj.append(ConformalTri(rc, rt, geod=geod))
        self.subproj = subproj
        self.bp = Barycentric(tgtpts)

    def transform(self, lon, lat):
        i = self.lune(lon, lat)
        mp = self.subproj[i]
        return mp.transform(lon, lat)

    def segment(self, x, y):
        bp = self.bp
        bary = bp.invtransform(x, y)
        return bp.segment(bary)

    def invtransform(self, x, y):
        i = self.segment(x, y)
        sp = self.subproj[i]
        return sp.invtransform(x, y)
