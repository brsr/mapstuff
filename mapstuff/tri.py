# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:46:33 2023

@author: brsr
"""
import numpy as np
import warnings
from scipy.optimize import minimize_scalar, root_scalar

from .projections import CtrlPtsProjection, KProjection, _unitsphgeod
from .transformations import UnitVector, Barycentric
from .helper import slerp, sqrt, normalize, central_angle, triangle_solid_angle

TGTPTS3 = np.eye(3)

class BarycentricMapProjection(CtrlPtsProjection):
    nctrlpts = 3
    tweak = False
    bcenter = np.ones(3)/3

    def fixbary(self, bary):
        if self.tweak:
            return self.fixbary_normalize(bary)
        else:
            return self.fixbary_subtract(bary)

    @staticmethod
    def fixbary_normalize(bary):
        """Converts array bary to an array with sum = 1 by dividing by
        bary.sum(). Will return nan if bary.sum() == 0.

        >>> fixbary_normalize(np.arange(3))
        array([0.        , 0.33333333, 0.66666667])
        """
        bary = np.array(bary)
        return bary / np.sum(bary, axis=0, keepdims=True)

    @staticmethod
    def fixbary_subtract(bary):
        """Converts array bary to an array with sum = 1 by subtracting
        (bary.sum() - 1)/bary.shape[0].

        >>> fixbary_subtract(np.arange(3))
        array([-0.66666667,  0.33333333,  1.33333333])
        """
        bary = np.array(bary)
        s = (np.sum(bary, axis=0, keepdims=True) - 1)/bary.shape[0]
        return bary - s

    def _fix_corners(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        #print(lon, lat, ctrlpts, result)
        #print(index0.shape, result.shape, np.array([1, 0, 0])[..., np.newaxis].shape)
        result[..., index0] = np.array([1, 0, 0])[..., np.newaxis]
        result[..., index1] = np.array([0, 1, 0])[..., np.newaxis]
        result[..., index2] = np.array([0, 0, 1])[..., np.newaxis]
        return result

    def _fix_corners_inv(self, bary, result):
        index0 = (bary[0] == 1)
        index1 = (bary[1] == 1)
        index2 = (bary[2] == 1)
        if np.any(index0):
            result[..., index0] = self.ctrlpts_v[..., 0, np.newaxis]
        if np.any(index1):
            result[..., index1] = self.ctrlpts_v[..., 1, np.newaxis]
        if np.any(index2):
            result[..., index2] = self.ctrlpts_v[..., 2, np.newaxis]
        return result
class Alfredo(BarycentricMapProjection):
    """this doesn't really accomplish anything. not triangular?"""

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        self.cosADfactor = (np.cross(np.roll(ctrlpts_v, 1, axis=1),
                            np.roll(ctrlpts_v, -1, axis=1), axis=0) +
                            ctrlpts_v * np.linalg.det(ctrlpts_v))
        self.tweak = tweak

    def transform_v(self, ll):
        rll = ll.reshape(2, -1)
        ctrlpts_v = self.ctrlpts_v
        cosADfactor = self.cosADfactor
        vtestpt = UnitVector.transform_v(rll)
        cosAPi = (vtestpt.T @ ctrlpts_v).T
        cosADi = (vtestpt.T @ cosADfactor).T
        pli = np.sqrt((1-cosAPi)/(1-cosADi))
        b = 1 - pli
        result = self.fixbary(b)
        shape = (3,) + ll.shape[1:]
        return result.reshape(shape)

    def invtransform(self, *args, **kwargs):
        return NotImplemented

class Fuller(BarycentricMapProjection):

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.tweak = tweak

    def transform(self, lon, lat):
        lon + 0#will TypeError if lon is not a number
        vtestpt = UnitVector.transform(lon, lat)
        ctrlpts_v = self.ctrlpts_v
        b = []
        for i in range(3):
            v0 = ctrlpts_v[..., i]
            v1 = ctrlpts_v[..., (i+1)%3]
            v2 = ctrlpts_v[..., (i+2)%3]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt20 = vtestpt @ np.cross(v2, v0)
            cosw01 = v0 @ v1
            cosw02 = v0 @ v2
            w01 = np.arccos(cosw01)
            w02 = np.arccos(cosw02)
            if np.isclose(w01, w02):
                w = (w01 + w02) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                g = vt12 + cosw*(vt01 + vt20)
                ti = self._b_eq(w, sinw, vt20, vt01, g)
            else:
                sinw01 = sqrt(1 - cosw01**2)
                sinw02 = sqrt(1 - cosw02**2)
                g = vt12 + cosw02*vt01 + cosw01*vt20
                ti = self._b_neq(w01, sinw02, vt01, w02, sinw01, vt20, g)
            b.append(1-ti)
        return self.fixbary(b)

    def _b_neq(self, w01, sinw02, vt01, w02, sinw01, vt20, g):
        t0 = (w01*sinw02*vt01 + w02*sinw01*vt20)/(g*w01*w02)
        if ~np.isfinite(t0):
            t0 = 0
        else:
            lim = np.pi/np.array([w01,w02]).max()
            t0 = np.clip(t0, -lim, lim)

        if abs(t0) < 1E-3:
            return t0
        w = (w01 + w02) / 2
        sinw = np.sin(w)
        t1 = self._b_eq(w, sinw, vt20, vt01, g)
        t0 = np.clip(t0, -abs(t1), abs(t1))
        c1 = sqrt(g**2 + (sinw01*vt20 - sinw02*vt01)**2)
        c2 = sqrt(g**2 + (sinw01*vt20 + sinw02*vt01)**2)
        d1 = np.arctan2(sinw01*vt20 - sinw02*vt01, g)
        d2 = np.arctan2(sinw01*vt20 + sinw02*vt01, g)
        def objective(t):
            if t < -lim or t > lim:
                return t**2, 2*t
            if t == 0:
                t = np.finfo(float).eps
            z = c1*np.cos((w01 - w02)*t - d1) - c2*np.cos((w01 + w02)*t - d2)
            dz = (-c1*(w01 - w02)*np.sin((w01 - w02)*t - d1)
                  + c2*(w01 + w02)*np.sin((w01 + w02)*t - d2))
            return z/t, (t*dz - z)*t**-2
        res = root_scalar(objective, fprime=True, method='newton', x0=t0)
        return res.root

    def _b_eq(self, w, sinw, vt20, vt01, gx):
        #x = sinw*(vt20 + vt01)/gx
        tx = np.arctan2(sinw*(vt20 + vt01),gx)/w
        #this form would be more efficient:
        #b = np.arctan2(sinw*vt12, cosw*vt12 + vt01 + vt20)/w
        return tx

    def invtransform(self, b1, b2, b3):
        if self.tweak:
            return self._invtransform_normalize(b1, b2, b3)
        else:
            return self._invtransform_subtract(b1, b2, b3)

    def _invtransform_subtract(self, b1, b2, b3):
        b1 + 0#will TypeError if not a number
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        def objective(k):
            f0 = np.where(bi[0] + k > 1, -1, 1)
            f1 = np.where(bi[1] + k > 1, -1, 1)
            f2 = np.where(bi[2] + k > 1, -1, 1)
            v01 = slerp(v1, v0, bi[0] + k)
            v02 = slerp(v2, v0, bi[0] + k)
            cx12 = np.cross(v01, v02)*f0
            v12 = slerp(v2, v1, bi[1] + k)
            v10 = slerp(v0, v1, bi[1] + k)
            cx20 = np.cross(v12, v10)*f1
            v20 = slerp(v0, v2, bi[2] + k)
            v21 = slerp(v1, v2, bi[2] + k)
            cx01 = np.cross(v20, v21)*f2

            v0x = normalize(np.cross(cx20, cx01))
            v1x = normalize(np.cross(cx01, cx12))
            v2x = normalize(np.cross(cx12, cx20))
            #this is slightly more robust than the triple product
            return (np.linalg.norm(v0x-v1x)
                    + np.linalg.norm(v1x-v2x)
                    + np.linalg.norm(v2x-v0x))
            # dv01 = dslerp(v1, v0, bi[0] + k)
            # dv02 = dslerp(v2, v0, bi[0] + k)
            # dcx12 = (np.cross(dv01, v02) + np.cross(v01, dv02))*f0
            # dv12 = dslerp(v2, v1, bi[1] + k)
            # dv10 = dslerp(v0, v1, bi[1] + k)
            # dcx20 = (np.cross(dv12, v10) + np.cross(v12, dv10))*f1
            # dv20 = dslerp(v0, v2, bi[2] + k)
            # dv21 = dslerp(v1, v2, bi[2] + k)
            # dcx01 = (np.cross(dv20, v21) + np.cross(v20, dv21))*f2

            # derivative = dcx12 @ v0x + dcx20 @ v1x + dcx01 @ v2x
            # return cx12 @ v0x, derivative
        if b1 == 0 or b2 == 0 or b3 == 0:
            k = 0
        elif np.allclose(self.sides, np.roll(self.sides, 1)):
            kx = self._k_eq(b1, b2, b3)
            k = kx[2]#FIXME is 2 always the right one?
        else:
            #FIXME why is this so freakin slow
            res = minimize_scalar(objective, bracket=[0,0.1])
            k = res.x
        #f0 = np.where(bi[0] + k > 1, -1, 1)
        f1 = np.where(bi[1] + k > 1, -1, 1)
        f2 = np.where(bi[2] + k > 1, -1, 1)
        #v01 = slerp(v1, v0, bi[0] + k)
        #v02 = slerp(v2, v0, bi[0] + k)
        #cx12 = np.cross(v01, v02)*f0
        v12 = slerp(v2, v1, bi[1] + k)
        v10 = slerp(v0, v1, bi[1] + k)
        cx20 = np.cross(v12, v10)*f1
        v20 = slerp(v0, v2, bi[2] + k)
        v21 = slerp(v1, v2, bi[2] + k)
        cx01 = np.cross(v20, v21)*f2

        v0x = normalize(np.cross(cx20, cx01))
        #v1x = normalize(np.cross(cx01, cx12))
        #v2x = normalize(np.cross(cx12, cx20))
        v0x = self._fix_corners_inv(bi, v0x)
        return UnitVector.invtransform_v(v0x)

    def _k_eq(self, b1, b2, b3):
        w = self.ca.mean()
        bi = np.array([b1, b2, b3])
        cw = np.cos(w)
        #sw = np.sin(w)
        cbw = np.cos(bi*w)
        sbw = np.sin(bi*w)
        pcbw = np.product(cbw)
        psbw = np.product(sbw)
        scc = np.sum(sbw * np.roll(cbw, -1) * np.roll(cbw, 1))
        css = np.sum(cbw*np.roll(sbw, -1)*np.roll(sbw, 1))
        objw2 = np.array([2*pcbw - cw - 1,
                          2*scc,
                          3*pcbw + 3 - css,
                          2*psbw])
        rts = np.roots(objw2)
        return np.arctan(rts)/w

    def _invtransform_normalize(self, b1, b2, b3):
        b1 + 0#will TypeError if not a number
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        def objective(k):
            f0 = np.where(bi[0] * k > 1, -1, 1)
            f1 = np.where(bi[1] * k > 1, -1, 1)
            f2 = np.where(bi[2] * k > 1, -1, 1)
            v01 = slerp(v1, v0, bi[0] * k)
            v02 = slerp(v2, v0, bi[0] * k)
            cx12 = normalize(np.cross(v01, v02))*f0
            v12 = slerp(v2, v1, bi[1] * k)
            v10 = slerp(v0, v1, bi[1] * k)
            cx20 = normalize(np.cross(v12, v10))*f1
            v20 = slerp(v0, v2, bi[2] * k)
            v21 = slerp(v1, v2, bi[2] * k)
            cx01 = normalize(np.cross(v20, v21))*f2
            v0x = normalize(np.cross(cx20, cx01))
            v1x = normalize(np.cross(cx01, cx12))
            v2x = normalize(np.cross(cx12, cx20))
            #i think this is slightly more robust than the triple product
            return (np.linalg.norm(v0x-v1x)
                    + np.linalg.norm(v1x-v2x)
                    + np.linalg.norm(v2x-v0x))

        res = minimize_scalar(objective, bracket=[1,1.1])
        k = res.x
        #f0 = np.where(bi[0] * k > 1, -1, 1)
        f1 = np.where(bi[1] * k > 1, -1, 1)
        f2 = np.where(bi[2] * k > 1, -1, 1)

        v12 = slerp(v2, v1, bi[1] * k)
        v10 = slerp(v0, v1, bi[1] * k)
        cx20 = normalize(np.cross(v12, v10))*f1
        v20 = slerp(v0, v2, bi[2] * k)
        v21 = slerp(v1, v2, bi[2] * k)
        cx01 = normalize(np.cross(v20, v21))*f2

        v0x = normalize(np.cross(cx20, cx01))
        v0x = self._fix_corners_inv(bi, v0x)
        return UnitVector.invtransform_v(v0x)

class SnyderEA(BarycentricMapProjection):

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        v_0 = ctrlpts_v[..., 0]
        v_1 = ctrlpts_v[..., 1]
        v_2 = ctrlpts_v[..., 2]
        self.v_01 = v_0 @ v_1
        self.v_12 = v_1 @ v_2
        self.v_20 = v_2 @ v_0
        self.v_012 = np.linalg.det(ctrlpts_v)
        self.c = self.v_12
        self.c2 = self.c**2
        self.s2 = 1 - self.c2
        self.s = sqrt(self.s2)
        self.w = np.arccos(self.c)
        self.midpoint_v = v_1 + v_2
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        lona = list(ctrlpts[0,:2]) + [self.midpoint[0],]
        lata = list(ctrlpts[1,:2]) + [self.midpoint[1],]
        self.area01m, _ = self.geod.polygon_area_perimeter(lona, lata)

    def transform(self, lon, lat):
        lon + 0
        actrlpts = self.ctrlpts
        ctrlpts_v = self.ctrlpts_v
        area = self.area
        geod = self.geod
        vtestpt = UnitVector.transform(lon, lat)
        lproj = -np.cross(np.cross(ctrlpts_v[..., 1], ctrlpts_v[..., 2]),
                          np.cross(ctrlpts_v[..., 0], vtestpt))
        norm = np.linalg.norm(lproj, axis=0, keepdims=True)
        if norm != 0:
            lproj = lproj / norm
        lllproj = UnitVector.invtransform_v(lproj)
        cosAP = ctrlpts_v[..., 0] @ vtestpt
        cosAD = ctrlpts_v[..., 0] @ lproj
        pl = sqrt((1-cosAP)/(1-cosAD))
        b0 = 1 - pl
        lona = [actrlpts[0,0], self.midpoint[0], lllproj[0]]
        lata = [actrlpts[1,0], self.midpoint[1], lllproj[1]]
        a1, _ = geod.polygon_area_perimeter(lona, lata)
        a1 += self.area01m
        b2 = a1/area * pl
        b1 = 1 - b0 - b2
        result = np.stack([b0,b1,b2])
        bresult = self._fix_corners(lon, lat, result)
        return np.where(np.isfinite(bresult), bresult, 0)

    def invtransform(self, b1, b2, b3):
        ctrlpts_v = self.ctrlpts_v
        area = self.area
        lp = np.array(1-b1)
        #make this an array so it won't complain about zero division, impute later
        a = b3/lp
        v_01 = self.v_01
        v_20 = self.v_20
        v_012 = self.v_012
        c = self.c
        s = self.s
        w = self.w
        Ar = a * area
        sA = np.sin(Ar)
        cA = 1 - np.cos(Ar)
        Fp = ((sA * v_012 + cA*(v_01*c - v_20))**2 - (s*cA*(1 + v_01))**2)
        Gp = 2*cA*s*(1 + v_01)*(sA*v_012 + cA*(v_01*c - v_20))
        result = 1/w*np.arctan2(Gp, Fp)
        vd = slerp(ctrlpts_v[..., 1], ctrlpts_v[..., 2], result)
        AdotD = ctrlpts_v[..., 0] @ vd
        AdotP = 1 - lp**2*(1-AdotD)
        t = np.arccos(AdotP)/np.arccos(AdotD)
        vresult = slerp(ctrlpts_v[..., 0], vd, t)
        bary = np.stack([b1, b2, b3])
        vresult = self._fix_corners_inv(bary, vresult)
        vresult[~np.isfinite(vresult)] = 0
        return UnitVector.invtransform_v(vresult)

class SnyderEA3(BarycentricMapProjection):
    tmat = np.array([[1/3,0,0],
                     [1/3,1,0],
                     [1/3,0,1]])
    tmatinv = np.array([[3,0,0],
                        [-1,1,0],
                        [-1,0,1]])

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        subproj = []
        #want the center that divides the triangle into 3 equal-area triangles
        ap = Areal(ctrlpts)
        center = ap.invtransform(1/3, 1/3, 1/3)
        self.center = center
        self.center_v = UnitVector.transform(*center)
        arr = np.arange(3)
        for i in range(3):
            index = np.roll(arr, -i)[1:]
            cp = np.concatenate([center[:,np.newaxis],
                                 ctrlpts[:, index]], axis=1)
            pj = SnyderEA(cp)
            subproj.append(pj)
        self.subproj = subproj

    def transform(self, lon, lat):
        subproj = self.subproj
        i = self.lune(lon, lat)
        pj = subproj[i-1]#shift because we want the opposite vertex
        betap = pj.transform(lon, lat)
        betax = self.tmat @ betap
        beta = np.roll(betax, i-1, axis=0)
        return beta

    def invtransform(self, b1, b2, b3):
        bary = np.array([b1,b2,b3])
        i = (Barycentric.segment(bary) ) % 3
        betax = np.roll(bary, -i, axis=0)
        betap = self.tmatinv @ betax
        pj = self.subproj[i]#FIXME ?
        return pj.invtransform(*betap)

class Areal(BarycentricMapProjection):
    """Spherical areal projection."""

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod: a pyproj.Geod object. For a unit sphere use
                pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        a_i = np.sum(np.roll(self.ctrlpts_v, -1, axis=1) *
                          np.roll(self.ctrlpts_v, 1, axis=1), axis=0)
        self.a_i = a_i
        self.b_i = (np.roll(a_i, -1) + np.roll(a_i, 1))/(1+a_i)
        self.tau_c = self.tau(self.area)

    def tau(self, area):
        """Convert areas on the geod to tau values for inverse transform"""
        return np.tan(area/self.totalarea*2*np.pi)

    def transform(self, x, y):
        try:
            areas = []
            for i in range(3):
                smtri = self.ctrlpts.copy()
                smtri[:, i] = np.array([x,y])
                a, _ = self.geod.polygon_area_perimeter(smtri[0],
                                                         smtri[1])
                areas.append(a)
            areas = np.array(areas)
            return areas/self.area
        except ValueError:
            raise TypeError()

    def invtransform_v(self, bary):
        rbary = bary.reshape(3,-1)
        if not self.geod.sphere:
            warnings.warn('inverse transform is approximate on ellipsoids')
        b_i = self.b_i[:,np.newaxis]
        tau = self.tau_c
        tau_i = self.tau(self.area*rbary)
        t_i = tau_i/tau
        c_i = t_i / ((1+b_i) + (1-b_i) * t_i)
        f_i = c_i / (1 - np.sum(c_i, axis=0))
        vector = self.ctrlpts_v @ f_i
        shape = [2] + list(bary.shape[1:])
        result = UnitVector.invtransform_v(vector).reshape(shape)
        return result
    
class Trilinear(BarycentricMapProjection):
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        vctrlpts = self.ctrlpts_v
        vxs = np.cross(np.roll(vctrlpts, -1, axis=1), 
                       np.roll(vctrlpts, 1, axis=1), axis=0)
        nvxs = np.linalg.norm(vxs, axis=0, keepdims=True)
        vxs /= nvxs
        self.vmatrix = vxs
        vm = vctrlpts * nvxs / np.linalg.det(vctrlpts)
        vmvm = vm.T @ vm #symmetric for a regular triangle, only 2 values
        self.invmatrix = vm
        self.newtonmatrix = vmvm
    
    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        vxs = self.vmatrix
        sintl = vxs.T @ vtestpt
        tl = np.arcsin(sintl)*self.sides
        beta = tl/tl.sum()        
        return self._fix_corners(lon, lat, beta)    
    
    def invtransform(self, b1, b2, b3, n=20, stop=1E-8):
        b1 + 0
        beta = np.array([b1,b2,b3])        
        vm = self.invmatrix
        vmvm = self.newtonmatrix
        h=1
        for i in range(n):
            s = np.sin(beta*h)
            c = np.cos(beta*h)
            f = np.einsum('i...,ij,j...', s, vmvm, s) - 1
            fprime = np.einsum('i...,ij,j...', s, vmvm, c)
            delta = -f/fprime
            h += delta
            print('delta:', delta)
            print('h: ', h)
            if np.max(np.abs(delta)) < stop:
                break
        s = np.sin(beta*h)
        v = vm @ s
        return UnitVector.invtransform_v(v)
#%%
class SplitAreaTri(BarycentricMapProjection):
    """Inverse is only approximate
    """
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        self.ctrldet = np.linalg.det(ctrlpts_v)
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        aream = []
        for i in range(3):
            am = triangle_solid_angle(ctrlpts_v[:,i], ctrlpts_v[:,(i+1)%3],
                                      midpoint_v[:,i])
            aream.append(am)
        self.aream = np.array(aream)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        areas = []
        vctrlpts = self.ctrlpts_v
        area = self.area
        dv = self.ctrldet
        for n in range(3):
            mi = self.midpoint_v[:,n-2]#?
            vx = np.cross(vctrlpts[..., n-1], vctrlpts[..., n])
            lproj = (vtestpt * dv - vctrlpts[:,n-2] * (vx @ vtestpt))
            a1 = triangle_solid_angle(vctrlpts[:,n-2], mi, lproj)
            areas.append(a1)
        areas = np.array(areas) + self.aream
        aa = areas/area
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, -i, axis=0)
            b = x*(1 - y + y*z)
            bx.append(b)        
        bx = np.array(bx)
        betax = bx/bx.sum()
        return self._fix_corners(lon, lat, betax)

class SplitLengthTri(BarycentricMapProjection):
    """Inverse is only approximate"""
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        self.ctrldet = np.linalg.det(ctrlpts_v)
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        
    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        aa = []
        vctrlpts = self.ctrlpts_v
        actrlpts = self.ctrlpts
        for n in range(3):#i = 2-n
            vc = np.roll(vctrlpts, 2-n, axis=1)
            ac = np.roll(actrlpts, 2-n, axis=1)
            mi = self.midpoint[:,n-2]
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            lllproj = UnitVector.invtransform_v(lproj)
            dist1x = central_angle(vc[..., 1], lproj)
            f, b, dist1x = self.geod.inv(mi[0], mi[1],
                                         lllproj[0],lllproj[1])
            f0, b0, _ = self.geod.inv(mi[0], mi[1],
                                      ac[0,2], ac[1,2])
            deltaf = (f-f0) % 360
            if (deltaf <= 90) | (deltaf > 270):
                s = 1
            else:
                s = -1
            t = s*dist1x/self.sides[n] + 1/2
            aa.append(t)
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, -i, axis=0)
            b = x*(1 - y + y*z)
            bx.append(b)
        bx = np.array(bx)
        betax = bx/bx.sum()     
        return self._fix_corners(lon, lat, betax)

class SplitAngleTri(BarycentricMapProjection):
    """Inverse is only approximate"""

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        tt = []
        vctrlpts = self.ctrlpts_v
        ctrl_angles = self.ctrl_angles/180*np.pi
        for n in range(3):
            vc = np.roll(vctrlpts, 2-n, axis=1)
            tantheta = (np.cross(vc[:,0], vc[:,1]) @ vtestpt)/(
                (vc[:,1] @ vtestpt) - (vc[:,0] @ vc[:,1]) * (vc[:,0] @ vtestpt)
                )
            tt.append(tantheta)
        thetas = np.arctan(tt)
        thetas[thetas < 0] += np.pi
        aa = thetas/ctrl_angles
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, -i, axis=0)
            b = x*(1 - y + y*z)
            bx.append(b)
        bx = np.array(bx)
        betax = bx/bx.sum()        
        return self._fix_corners(lon, lat, betax)
#%%
class BisectTri(BarycentricMapProjection):
    """Inverse is only approximate
    """
    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        self.ctrldet = np.linalg.det(ctrlpts_v)
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)
        aream = []
        for i in range(3):
            am = triangle_solid_angle(ctrlpts_v[:,i], ctrlpts_v[:,(i+1)%3],
                                      midpoint_v[:,i])
            #vc[:,0], mi, lproj)
            aream.append(am)
        self.aream = np.array(aream)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        areas = []
        vctrlpts = self.ctrlpts_v
        #actrlpts = self.ctrlpts
        #geod = self.geod
        area = self.area
        for i in range(3):
            vc = np.roll(vctrlpts, i, axis=1)
            #ac = np.roll(actrlpts, i, axis=1)
            mi = self.midpoint_v[:,-i]#?
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            #lllproj = UnitVector.invtransform_v(lproj)
            #loni = [ac[0,0], mi[0], lllproj[0]]
            #lati = [ac[1,0], mi[1], lllproj[1]]
            #a1, _ = geod.polygon_area_perimeter(loni, lati)
            a1 = triangle_solid_angle(vc[:,0], mi, lproj)
            areas.append(a1)
        areas = np.array(areas) + self.aream
        aa = areas/area
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, i, axis=0)
            b = (y**2 * x**2 + z**2 * x**2 - y**2 * z**2
                - x * y**2 + z * y**2
                - 2*y*x**2 - x*z**2 + y*z**2 + x**2
                + 3*y*x + z*x - 2*y*z
                - 2*x - y + z + 1)
            bx.append(b)    
        bx = np.array(bx)
        betax = bx/bx.sum()
        return self._fix_corners(lon, lat, betax)

    def invtransform(self, b1, b2, b3):
        b1 + 0
        beta = np.array([b1,b2,b3])
        vctrlpts3 = self.ctrlpts_v

        #xs = []
        ptts = []
        for i in range(3):
            beta1, beta2, beta3 = np.roll(beta, -i, axis=0)
            x = beta2/(1 - beta1)
            #xs.append(x)
            a = x * self.area
            pt0 = vctrlpts3[:,i]
            pt1 = vctrlpts3[:,i-2]
            pt2 = vctrlpts3[:,i-1]
            cosw = pt1 @ pt2
            w = np.arccos(cosw)
            sinw = np.sin(w)
            p2 = ((np.cos(a/2)* pt2 @ np.cross(pt0, pt1)- np.sin(a/2)*pt2 @ (pt1 + pt0))
                  + np.sin(a/2)*cosw*(1 + pt1 @ pt0))
            p3 = sinw*np.sin(a/2)*(1 + pt0 @ pt1)
            r = 2*p3*p2/(p2**2 - p3**2)
            t = np.arctan(r)/w#really close to just x
            #print(x, t)
            #t = x
            ptt = slerp(pt2, pt1, t)
            ptts.append(ptt)

        ptts = np.array(ptts).T
        ns = np.cross(vctrlpts3, ptts, axis=0)
        pts = np.cross(ns, np.roll(ns, -1, axis=1), axis=0)
        v = pts.sum(axis=1)
        v = self._fix_corners_inv(beta, v)
        return UnitVector.invtransform_v(v)

class BisectTri2(BarycentricMapProjection):
    """Inverse is only approximate"""

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        #v_0 = ctrlpts_v[..., 0]
        #v_1 = ctrlpts_v[..., 1]
        #v_2 = ctrlpts_v[..., 2]
        midpoint_v = np.roll(ctrlpts_v, 1, axis=1) + np.roll(ctrlpts_v, -1, axis=1)
        midpoint_v /= np.linalg.norm(midpoint_v, axis=0, keepdims=True)
        self.midpoint_v = midpoint_v
        self.midpoint = UnitVector.invtransform_v(self.midpoint_v)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        aa = []
        vctrlpts = self.ctrlpts_v
        actrlpts = self.ctrlpts
        for i in range(3):
            vc = np.roll(vctrlpts, i, axis=1)
            ac = np.roll(actrlpts, i, axis=1)
            mi = self.midpoint[:,-i]
            lproj = -np.cross(np.cross(vc[..., 1], vc[..., 2]),
                              np.cross(vc[..., 0], vtestpt))
            lllproj = UnitVector.invtransform_v(lproj)
            dist1x = central_angle(vc[..., 1], lproj)
            f, b, dist1x = self.geod.inv(mi[0], mi[1],
                                         lllproj[0],lllproj[1])
            f0, b0, _ = self.geod.inv(mi[0], mi[1],
                                      ac[0,2], ac[1,2])
            deltaf = (f-f0) % 360
            if (deltaf <= 90) | (deltaf > 270):
                s = 1
            else:
                s = -1
            t = s*dist1x/self.sides[i] + 1/2
            #print(t)
            aa.append(t)
        bx = []
        for i in range(3):
            x,y,z = np.roll(aa, i, axis=0)
            b = (y**2 * x**2 + z**2 * x**2 - y**2 * z**2
                - x * y**2 + z * y**2
                - 2*y*x**2 - x*z**2 + y*z**2 + x**2
                + 3*y*x + z*x - 2*y*z
                - 2*x - y + z + 1)
            bx.append(b)
        bx = np.array(bx)
        betax = bx/bx.sum()
        return self._fix_corners(lon, lat, betax)

    def invtransform(self, b1, b2, b3):
        b1 + 0
        beta = np.array([b1,b2,b3])
        vctrlpts3 = self.ctrlpts_v

        #xs = []
        ptts = []
        for i in range(3):
            beta1, beta2, beta3 = np.roll(beta, -i, axis=0)
            x = beta2/(1 - beta1)
            pt1 = vctrlpts3[:,i-2]
            pt2 = vctrlpts3[:,i-1]
            ptt = slerp(pt2, pt1, x)
            ptts.append(ptt)

        ptts = np.array(ptts).T
        ns = np.cross(vctrlpts3, ptts, axis=0)
        pts = np.cross(ns, np.roll(ns, -1, axis=1), axis=0)
        v = pts.sum(axis=1)
        v = self._fix_corners_inv(beta, v)
        return UnitVector.invtransform_v(v)

class FullerEq(BarycentricMapProjection):

    def transform_v(self, ll):
        vtestpt_pre = UnitVector.transform(*ll)
        vtestpt = vtestpt_pre.reshape(3,-1)
        ctrlpts_v = self.ctrlpts_v
        b = []
        for i in range(3):
            v0 = ctrlpts_v[..., i]
            v1 = ctrlpts_v[..., (i+1)%3]
            v2 = ctrlpts_v[..., (i-1)%3]
            cosw01 = v0 @ v1
            cosw02 = v0 @ v2
            w01 = np.arccos(cosw01)
            w02 = np.arccos(cosw02)
            w = (w01 + w02) / 2
            sinw = np.sin(w)
            cosw = np.cos(w)
            vt01 = np.tensordot(vtestpt, np.cross(v0, v1), axes=(0,0))
            vt12 = np.tensordot(vtestpt, np.cross(v1, v2), axes=(0,0))
            vt20 = np.tensordot(vtestpt, np.cross(v2, v0), axes=(0,0))
            bi = np.arctan2(sinw*vt12, cosw*vt12 + vt01 + vt20)/w
            #gx = vt12 + cosw*(vt01 + vt20)
            #tx = np.arctan2(sinw*(vt20 + vt01),gx)/w
            b.append(bi)
            #b.append(1-tx)
        b = np.array(b)
        result = self.fixbary_subtract(b)
        return result.reshape(vtestpt_pre.shape)

    def invtransform(self, b1, b2, b3):
        b1 + 0 #still not vectorized
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]

        w = self.ca.mean()
        bi = np.array([b1, b2, b3])
        cw = np.cos(w)
        #sw = np.sin(w)
        cbw = np.cos(bi*w)
        sbw = np.sin(bi*w)
        pcbw = np.product(cbw)
        psbw = np.product(sbw)
        scc = np.sum(sbw * np.roll(cbw, -1) * np.roll(cbw, 1))
        css = np.sum(cbw*np.roll(sbw, -1)*np.roll(sbw, 1))
        objw2 = np.array([2*pcbw - cw - 1,
                          2*scc,
                          3*pcbw + 3 - css,
                          2*psbw])
        rts = np.roots(objw2)[-1]#FIXME solve this cubic explicitly
        rts = rts.real
        k = np.arctan(rts)/w
        #f0 = np.where(bi[0] + k > 1, -1, 1)
        f1 = np.where(bi[1] + k > 1, -1, 1)
        f2 = np.where(bi[2] + k > 1, -1, 1)
        #v01 = slerp(v1, v0, bi[0] + k)
        #v02 = slerp(v2, v0, bi[0] + k)
        #cx12 = np.cross(v01, v02)*f0
        v12 = slerp(v2, v1, bi[1] + k)
        v10 = slerp(v0, v1, bi[1] + k)
        cx20 = np.cross(v12, v10)*f1
        v20 = slerp(v0, v2, bi[2] + k)
        v21 = slerp(v1, v2, bi[2] + k)
        cx01 = np.cross(v20, v21)*f2

        v0x = normalize(np.cross(cx20, cx01))
        #v1x = normalize(np.cross(cx01, cx12))
        #v2x = normalize(np.cross(cx12, cx20))
        v0x = self._fix_corners_inv(bi, v0x)
        #print(v0x)
        return UnitVector.invtransform_v(v0x)

class SnyderEASym(BarycentricMapProjection):

    def __init__(self, ctrlpts, p = SnyderEA):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        subproj = []
        for i in range(3):
            cp = np.roll(ctrlpts, i, axis=1)
            pj = p(cp)
            subproj.append(pj)
        self.subproj = subproj

    def transform(self, lon, lat):
        subproj = self.subproj
        for i in range(3):
            pj = subproj[i]
            b = np.roll(pj.transform(lon, lat), -i, axis=0)
            try:
                beta += b
            except NameError:
                beta = b
        return beta/3

    def invtransform(self, *args, **kwargs):
        subproj = self.subproj
        for i in range(3):
            pj = subproj[i]
            b = np.roll(args, i, axis=0)
            r = UnitVector.transform(pj.invtransform(b))
            try:
                result += r
            except NameError:
                result = r
        return UnitVector.invtransform(normalize(result))

class ReverseFuller(BarycentricMapProjection):

    def __init__(self, ctrlpts, tweak=False):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.tweak = tweak

    def transform(self, *args, **kwargs):
        return NotImplemented

    def invtransform(self, b1, b2, b3):
        bi = np.array([b1, b2, b3])
        v0 = self.ctrlpts_v[..., 0]
        v1 = self.ctrlpts_v[..., 1]
        v2 = self.ctrlpts_v[..., 2]
        v01 = slerp(v1, v0, b1)
        v02 = slerp(v2, v0, b1)
        cx12 = normalize(np.cross(v01, v02))
        v12 = slerp(v2, v1, b2)
        v10 = slerp(v0, v1, b2)
        cx20 = normalize(np.cross(v12, v10))
        v20 = slerp(v0, v2, b3)
        v21 = slerp(v1, v2, b3)
        cx01 = normalize(np.cross(v20, v21))

        v0x = np.cross(cx20, cx01)
        v1x = np.cross(cx01, cx12)
        v2x = np.cross(cx12, cx20)
        vx = np.stack([v0x,v1x,v2x], axis=-1)
        if not self.tweak:
            vx = normalize(vx)
        result = vx.mean(axis=-1)
        result = self._fix_corners_inv(bi, result)
        return UnitVector.invtransform_v(result)

class NSlerpTri(BarycentricMapProjection, KProjection):

    def __init__(self, ctrlpts, k=1, exact=True, pow=1, eps=0):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        self.pow = pow
        angles = self.sides
        self.eq = (np.max(angles) - np.min(angles)) <= eps
        if self.eq:
            self.avangle = np.mean(angles)

    def transform(self, *args, **kwargs):
        return NotImplemented

    def _tri_naive_slerp_angles(self, bary):
        """Interpolates the angle factor so that it's equal to the
        angle between pts 1 and 2 when beta_3=0, etc.
        """
        angles = self.sides
        if self.eq:
            return self.avangle
        a = bary[0]
        b = bary[1]
        c = bary[2]
        ab = (a*b)**self.pow
        bc = (b*c)**self.pow
        ca = (c*a)**self.pow
        denom = ab + bc + ca
        numer = ab*angles[0] + bc*angles[1] + ca*angles[2]
        return numer/denom

    def invtransform_v(self, bary):
        base = self.ctrlpts_v
        angles = self._tri_naive_slerp_angles(bary)
        b = np.sin(angles * bary) / np.sin(angles)
        result = (b.T.dot(base.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(bary, result)
        return UnitVector.invtransform_v(result)
#%%
class SmallCircleEA(BarycentricMapProjection):

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        ctrlpts_v = self.ctrlpts_v
        v_0 = ctrlpts_v[..., 0]
        v_1 = ctrlpts_v[..., 1]
        v_2 = ctrlpts_v[..., 2]
        self.v_01 = v_0 @ v_1
        self.v_12 = v_1 @ v_2
        self.v_20 = v_2 @ v_0
        self.v_012 = np.linalg.det(ctrlpts_v)
        self.c = self.v_12
        self.c2 = self.c**2
        self.s2 = 1 - self.c2
        self.s = sqrt(self.s2)
        self.w = np.arccos(self.c)
        cBC = np.cross(v_1, v_2)
        cBC /= np.linalg.norm(cBC)
        self.vD = cBC
        self.vCxD = np.cross(v_2, self.vD)
        beta1 = np.arctan2(self.v_012, ( v_0@v_2 - (v_1 @ v_0)*(v_1 @ v_2)))
        beta2 = np.arctan2(self.v_012, (v_0@v_1 - (v_2 @ v_0)*(v_1 @ v_2)))
        gamma = np.arctan2(self.v_012, (v_1@v_2 - (v_0 @ v_1)*(v_2 @ v_0)))
        self.beta1 = beta1#np.where(beta1 < 0, beta1 + np.pi, beta1)
        self.beta2 = beta2#np.where(beta2 < 0, beta2 + np.pi, beta2)
        self.gamma = gamma#np.where(gamma < 0, gamma + np.pi, gamma)
        self.area = self.beta1 + self.beta2 + self.gamma - np.pi

    def transform(self, lon, lat):
        lon + 0
        ctrlpts_v = self.ctrlpts_v
        vC = ctrlpts_v[..., 2]
        vD = self.vD
        vCxD = self.vCxD
        vP = UnitVector.transform(lon, lat)
        PD = np.arccos(vP @ vD)
        BC = np.arccos(self.v_12)
        b = np.pi/2 - PD

        beta1 = self.beta1#ok
        beta2 = self.beta2#ok
        gamma = self.gamma#ok

        b = np.pi/2 - PD  # = np.arcsin (vP @ vD)
        phi1 = np.arcsin(np.clip(np.cos(beta1)/np.cos(b), -1, 1))
        phi2 = np.arcsin(np.clip(np.cos(beta2)/np.cos(b), -1, 1))
        f1 = np.arcsin(np.clip(np.tan(b)/np.tan(beta1), -1, 1))
        f2 = np.arcsin(np.clip(np.tan(b)/np.tan(beta2), -1, 1))
        d = BC - f1 - f2

        at = self.area
        nota = gamma - phi1 - phi2 - d * np.sin(b)
        u = np.sqrt(np.clip(nota/at,0,np.inf))

        psi = np.arctan2(vP @ vCxD, vP @ vC ) # angle PDC

        #psi = np.arccos(np.clip((vP@vC - (vD @ vP)*(vD @ vC))/(
        #    np.linalg.norm(np.cross(vD, vP))*np.linalg.norm(np.cross(vD,vC))),-1,1))
        #psi = np.where(psi < 0, psi + np.pi, psi)
        v = (psi - f2)/d
        b0 = 1 - u
        b1 = v*u
        b2 = 1 - b0 - b1
        result = np.stack([b0,b1,b2])
        bresult = self._fix_corners(lon, lat, result)
        return np.where(np.isfinite(bresult), bresult, 0)
    
class TriGnomonic(BarycentricMapProjection):

    def __init__(self, ctrlpts):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.invv = np.linalg.pinv(self.ctrlpts_v)

    def transform(self, lon, lat):
        lon + 0
        vtestpt = UnitVector.transform(lon, lat)
        bary = self.invv @ vtestpt
        return self.fixbary_normalize(bary)

    def invtransform(self, *args):
        bary = np.array(args)
        result = self.ctrlpts_v @ bary
        return UnitVector.invtransform(normalize(result))    