# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:38:02 2019

@author: brsr
"""
import pyproj
import warnings
import numpy as np
from abc import ABC
from scipy.optimize import minimize

from .transformations import Transformation, UnitVector
from .helper import sqrt, antipode_v, central_angle, trigivenlengths, triangle_solid_angle

#TODO:
#vectorize all the things, or convert to 
#make a better implementation of conformal

#arange3 = np.arange(3)
#FIRST AXIS IS SPATIAL

_unitsphgeod = pyproj.Geod(a=1, b=1)

class CtrlPtsProjection(Transformation, ABC):
    """Subclass for any map projection that uses (2 or more) control points."""
    def __init__(self, ctrlpts, geod = _unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 or 2x4 Numpy array, latitude and longitude of
            each control point
        geod= a pyproj.Geod object. For a unit sphere use
            pyproj.Geod(a=1,b=1)
        """
        n = ctrlpts.shape[1]
        if self.nctrlpts != n:
            raise ValueError(
                'ctrlpts has wrong number of points for this projection')
        self.geod = geod
        #it's possible to get a geod where this would give the wrong answer,
        #but I think it would have to be really weird
        area, _ = geod.polygon_area_perimeter([0,120,-120],[0,0,0])
        self.totalarea = 2*area

        self.ctrlpts = ctrlpts
        ctrlpts_v = UnitVector.transform_v(ctrlpts)
        self.ctrlpts_v = ctrlpts_v
        center_v = ctrlpts_v.sum(axis=1)
        self.center_v = center_v / np.linalg.norm(center_v)
        self.center = UnitVector.invtransform_v(center_v)
        antipode = antipode_v(ctrlpts)
        self.antipode = antipode
        self.antipode_v = UnitVector.transform_v(antipode)
        self.sa = 0
        if self.nctrlpts > 2:
            faz, baz, sides = self.geod.inv(ctrlpts[0], ctrlpts[1],
                                              np.roll(ctrlpts[0], -1),
                                              np.roll(ctrlpts[1], -1))
            self.sides = sides
            self.faz = faz
            self.baz = baz
            self.ctrl_angles = (faz - np.roll(baz, 1))%360
            area, _ = geod.polygon_area_perimeter(*ctrlpts)
            self.area = area
            self.ca = central_angle(ctrlpts_v,
                                        np.roll(ctrlpts_v, -1, axis=1))
            for i in range(1, self.nctrlpts-1):
                self.sa += triangle_solid_angle(ctrlpts_v[..., 0],
                                                    ctrlpts_v[..., i],
                                                    ctrlpts_v[..., i+1])

            self.edgenormals = np.cross(ctrlpts_v,
                                        np.roll(ctrlpts_v, -1, axis=1), axis=0)

        else:
            faz, baz, sides = self.geod.inv(ctrlpts[0,0], ctrlpts[1,0],
                                              ctrlpts[0,1], ctrlpts[1,1])
            self.sides = sides
            self.faz = faz
            self.baz = baz
            self.area = 0
            self.ca = central_angle(ctrlpts_v[..., 0], ctrlpts_v[..., 1])
            self.edgenormals = np.cross(ctrlpts_v[..., 0], ctrlpts_v[..., 1])

        self.cosca = np.cos(self.ca)
        self.sinca = np.sin(self.ca)

        if self.sa < 0:
            warnings.warn('control polygon is in negative orientation, '
                          + 'may cause unusual results')

        if self.nctrlpts == 4:
            ctrlpts_v = self.ctrlpts_v
            v0 = ctrlpts_v[..., 0]
            v1 = ctrlpts_v[..., 1]
            v2 = ctrlpts_v[..., 2]
            v3 = ctrlpts_v[..., 3]
            poip1 = np.cross(np.cross(v0, v1), np.cross(v3, v2))
            poip2 = np.cross(np.cross(v0, v3), np.cross(v1, v2))
            poip = np.stack([[poip1, -poip1],
                             [poip2, -poip2]]).transpose(2,0,1)
            poip = poip / np.linalg.norm(poip, axis=0)
            self.poi_v = poip
            self.poi = UnitVector.invtransform_v(poip)
            self.crossx = np.cross(ctrlpts_v,
                                   np.roll(ctrlpts_v, -2, axis=1),
                                   axis=0)[..., :2]

    def orienttgtpts(self, tgtpts, N = (0, 90)):
        """Orient target points so that line from 0 to the projection of N
        points up. Will fail if map projection doesn't define tgtpts."""
        pN = self.transform(*N)
        if np.allclose(pN, [0,0]):
            raise ValueError('projection of N too close to 0')
        angle = np.arctan2(pN[0],pN[1])
        rotm = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
        result = rotm @ tgtpts
        self.tgtpts = result

    def lune(self, lon, lat):
        """
        Determine which lune a point or series of points lies in.
        Lune 0 is the lune with vertex at the centroid and edges passing through
        control points 0 and 1. Lune 1 is the same using control pts 1 and 2,
        and Lune 2 uses control pts 2 and 0.
        """
        #inexact on ellipsoids but close enough
        testpt = UnitVector.transform(lon, lat)
        testpt_v = testpt.reshape(3,-1)
        ctrlpts_v = self.ctrlpts_v
        center_v = self.center_v
        cx = np.cross(center_v, ctrlpts_v, axis=0)
        sk = cx.T @ testpt_v
        sg = sk >= 0
        ind = sg & ~np.roll(sg, shift=-1, axis=0)
        result = np.argmax(ind, axis=0)
        return result.reshape(testpt.shape[1:])

class Double(CtrlPtsProjection):
    """Linear combination of two projections"""
    def __init__(self, ctrlpts, proj1, proj2, t=0.5):
        subproj = [proj1(ctrlpts), proj2(ctrlpts)]
        self.nctrlpts = subproj[0].nctrlpts
        if self.nctrlpts != subproj[1].nctrlpts:
            raise ValueError('proj1 and proj2 have different # of ctrlpts')
        super().__init__(ctrlpts)
        self.subproj = subproj
        self.t = t

    def transform(self, lon, lat):
        subproj = self.subproj
        t = self.t
        return ((1 - t)*subproj[0].transform(lon, lat)
                + t*subproj[1].transform(lon, lat))

    def inv_transform(self, lon, lat):
        subproj = self.subproj
        t = self.t
        return ((1 - t)*subproj[0].transform(lon, lat)
                + t*subproj[1].transform(lon, lat))

class KProjection(CtrlPtsProjection):
    exact = True
    k = 1
    def extend(self, v):
        normal = self.center_v
        k = self.k
        n = np.linalg.norm(v, axis=0, keepdims=True)
        if self.exact:
            vdotc = np.tensordot(v, normal, axes=(0, 0))[np.newaxis]
            vdotv = n**2
            p = -vdotc + sqrt(1 + vdotc**2 - vdotv)
        else:
            p = 1 - n
        #print(v.shape, p.shape, normal.shape)
        return v + k*p*normal[..., np.newaxis]
    
    def transform(self, *args, **kwargs):
        return NotImplemented
    
#%% not-polygonal projections
class ChambTrimetric(CtrlPtsProjection):
    """Chamberlin trimetric projection"""
    #FIXME this implementation fails for control triangles with 
    #high aspect ratios
    nctrlpts = 3

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        super().__init__(ctrlpts, geod)
        self.tgtpts = trigivenlengths(self.sides)
        try:
            self.orienttgtpts(self.tgtpts)
        except ValueError:
            pass

    def transform(self, x, y, **kwargs):
        if hasattr(x, '__iter__'):
            raise TypeError()
        tgtpts = self.tgtpts
        f, b, rad = self.geod.inv(self.ctrlpts[0], self.ctrlpts[1],
                                   x*np.ones(3), y*np.ones(3))
        faz = self.faz
        raz1 = (faz - f) % 360
        radsq = np.array(rad).squeeze()**2
        ctgt = tgtpts.T.copy().view(dtype=complex).squeeze()
        a = np.roll(ctgt, -1) - ctgt
        b = ctgt
        l = abs(a)
        lsq = l**2
        rsq = radsq/lsq
        ssq = np.roll(radsq, -1, axis=-1)/lsq
        x0 = (rsq - ssq + 1)/2
        y0 = sqrt(-rsq**2 + 2*rsq*(ssq + 1) - (ssq - 1)**2)/2
        y0[np.isnan(y0)] = 0
        y = np.where(raz1 > 180, -y0, y0)
        z0 = x0 +1j*y
        pts = (a * z0 + b)
        result = np.mean(pts)
        return result.real, result.imag

    def invtransform(self, *args, **kwargs):
        return NotImplemented

class LstSqTrimetric(ChambTrimetric):
    """Least-squares variation of the Chamberlin trimetric projection"""
    def transform(self, x, y, **kwargs):
        init = super().transform(x, y)
        tgtpts = self.tgtpts
        f, b, rad = self.geod.inv(self.ctrlpts[0], self.ctrlpts[1],
                                   x*np.ones(3), y*np.ones(3))
        def objective(v):
            x = v[0]
            y = v[1]
            a = tgtpts[0]
            b = tgtpts[1]
            xma = x-a
            ymb = y-b
            dist = np.sqrt(xma**2 + ymb**2)
            result = np.sum((dist - rad)**2 )
            f = 1 - rad/dist
            f[rad <= 0] = 1
            jac = 2*np.array([np.sum(xma*f), np.sum(ymb*f)])
            return result, jac
        res = minimize(objective, init, jac=True,
                       method = 'BFGS')
        return res.x

class LinearTrimetric(CtrlPtsProjection):
    """The linear variation of the Chamberlin Trimetric projection."""
    nctrlpts = 3
    matrix1 = np.array([[0,-1],
               [1,0]])
    matrix2 = np.array([[0, -1, 1],
               [1, 0, -1],
               [-1, 1, 0]])
    matrixinv1 = np.array([[-2,1,1],
              [1,-2,1],
              [1,1,-2]])*2/3

    def __init__(self, ctrlpts, geod=_unitsphgeod):
        """Parameters:
        ctrlpts: 2x3 Numpy array, latitude and longitude of each control point
        geod= a pyproj.Geod object. For a unit sphere use
        pyproj.Geod(a=1,b=1).
        """
        super().__init__(ctrlpts, geod)
        self.radius = ((geod.a**(3/2) + geod.b**(3/2))/2)**(2/3)
        self.tgtpts = trigivenlengths(self.sides)
        self.setmat()
        # try:
        #     self.orienttgtpts(self.tgtpts)
        #     self.setmat()
        # except ValueError:
        #     pass

        vctrl = self.ctrlpts_v
        self.invctrlvector = np.linalg.pinv(vctrl)
        self.invperpmatrix = self.invctrlvector @ self.invctrlvector.T
        cosrthmin = 1 / np.sqrt(self.invperpmatrix.sum())
        self.hminall = np.arccos(cosrthmin)**2

    def setmat(self, tgtpts=None):
        """Set matrices that use tgtpts"""
        if tgtpts is None:
            tgtpts = self.tgtpts
        else:
            self.tgtpts = tgtpts
        tgtde = np.linalg.det(np.concatenate([tgtpts, np.ones((1,3))], axis=0))
        self.m = self.matrix1 @ tgtpts @ self.matrix2 /(2*tgtde)
        self.minv = self.matrixinv1 @ tgtpts.T

    def transform_v(self, pts):
        rpts = pts.reshape((2,-1)).T
        rad = []
        for x,y in rpts:
            f, b, radi = self.geod.inv(x*np.ones(3), y*np.ones(3),
                          self.ctrlpts[0], self.ctrlpts[1])
            rad.append(radi)
        shape = list(pts.shape)
        shape[0] = 3
        rad = np.array(rad).T
        radsq = np.array(rad)**2
        result = self.m @ radsq
        return result.reshape(pts.shape)

    def invtransform_v(self, pts, n=20, stop=1E-8):
        if not self.geod.sphere:
            warnings.warn('inverse transform is approximate on ellipsoids')
        rpts = pts.reshape((2,-1))
        k = self.minv @ rpts/self.radius**2
        hmin = -np.min(k, axis=0)
        print('k: ', k)
        #hmax = np.pi**2-np.max(k, axis=0)
        hminall = self.hminall
        h = np.where(hmin < hminall, hminall, hmin)
        print('h: ', h)
        for i in range(n):
            rsq = (k + h)
            #pos = rsq > 0
            neg = rsq < 0
            zer = rsq == 0
            c = np.where(neg, np.cosh(np.sqrt(-rsq)), np.cos(np.sqrt(rsq)))
            b = np.where(neg, np.sinh(np.sqrt(-rsq)),
                         np.sin(np.sqrt(rsq)))/np.sqrt(np.abs(rsq))
            b[zer] = 1
            f = np.einsum('i...,ij,j...', c, self.invperpmatrix, c) - 1
            fprime = np.einsum('i...,ij,j...', c, self.invperpmatrix, b)
            delta = f/fprime
            h += delta
            print('delta:', delta)
            print('h: ', h)
            if np.max(np.abs(delta)) < stop:
                break
        #h = np.clip(h, hmin, hmax)
        rsq = np.clip(k + h, 0, np.pi**2)
        c = np.cos(np.sqrt(rsq))
        vector = self.invctrlvector.T @ c
        print(c)
        print(vector)
        return UnitVector.invtransform_v(vector).reshape(pts.shape)

    def nmforplot(self, pts, n=100):
        rpts = pts.reshape((2,-1))
        k = self.minv @ rpts/self.radius**2
        hmin = -np.min(k, axis=0)
        hmax = np.pi**2-np.max(k, axis=0)
        h = np.linspace(hmin,hmax,100).T
        rsq = (k[..., np.newaxis] + h)
        c = np.cos(np.sqrt(rsq))
        nm = np.einsum('i...,ij,j...', c, self.invperpmatrix, c)

        return h, nm


#%%
if __name__ == "__main__":
    import doctest
    sup = np.testing.suppress_warnings()
    sup.filter(RuntimeWarning)
    options = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    with sup:
        doctest.testmod(optionflags = options)
