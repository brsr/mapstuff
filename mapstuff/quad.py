# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:44:44 2023

@author: brsr
"""
import numpy as np
from scipy.optimize import root_scalar 

from .projections import CtrlPtsProjection, KProjection
from .transformations import UnitVector, Barycentric
from .helper import slerp, sqrt
from .tri import SnyderEA

TGTPTS4 = np.array([[0, 1, 1, 0],
                    [0, 0, 1, 1]])

class UVMapProjection(CtrlPtsProjection):
    nctrlpts = 4
    bcenter = np.ones(2)/2

    def _fix_corners(self, lon, lat, result):
        ctrlpts = self.ctrlpts
        index0 = (lon == ctrlpts[0,0]) & (lat == ctrlpts[1,0])
        index1 = (lon == ctrlpts[0,1]) & (lat == ctrlpts[1,1])
        index2 = (lon == ctrlpts[0,2]) & (lat == ctrlpts[1,2])
        index3 = (lon == ctrlpts[0,3]) & (lat == ctrlpts[1,3])
        result[..., index0] = np.array([ 0,  0])[..., np.newaxis]
        result[..., index1] = np.array([ 1,  0])[..., np.newaxis]
        result[..., index2] = np.array([ 1,  1])[..., np.newaxis]
        result[..., index3] = np.array([ 0,  1])[..., np.newaxis]
        return result

    def _fix_corners_inv(self, x, y, result):
        index0 = (x == 0) & (y == 0)
        index1 = (x == 1) & (y == 0)
        index2 = (x == 1) & (y == 1)
        index3 = (x == 0) & (y == 1)
        if np.any(index0):
            result[..., index0] = self.ctrlpts_v[..., 0, np.newaxis]
        if np.any(index1):
            result[..., index1] = self.ctrlpts_v[..., 1, np.newaxis]
        if np.any(index2):
            result[..., index2] = self.ctrlpts_v[..., 2, np.newaxis]
        if np.any(index3):
            result[..., index3] = self.ctrlpts_v[..., 3, np.newaxis]
        return result
    

class CriderEq(UVMapProjection):
    def transform_v(self, ll):
        vtestpt = UnitVector.transform(*(ll.reshape(2,-1)))
        ctrlpts_v = self.ctrlpts_v
        result = []
        for p in [(0, 1, 2, 3),(1, 2, 3, 0)]:
            #FIXME can calculate a lot of this stuff beforehand
            v0 = ctrlpts_v[..., p[0]]
            v1 = ctrlpts_v[..., p[1]]
            v2 = ctrlpts_v[..., p[2]]
            v3 = ctrlpts_v[..., p[3]]
            cosw01 = v0 @ v1
            cosw23 = v2 @ v3
            w01 = np.arccos(cosw01)
            w23 = np.arccos(cosw23)
            #sinw01 = sqrt(1 - cosw01**2)
            #sinw23 = sqrt(1 - cosw23**2)
            w = (w01 + w23) / 2
            sinw = np.sin(w)
            cosw = np.cos(w)

            #vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = np.tensordot(vtestpt, np.cross(v1, v2), axes=(0,0))
            #vt23 = vtestpt @ np.cross(v2, v3)
            vt30 = np.tensordot(vtestpt, np.cross(v3, v0), axes=(0,0))
            vt02 = np.tensordot(vtestpt, np.cross(v0, v2), axes=(0,0))
            vt13 = np.tensordot(vtestpt, np.cross(v1, v3), axes=(0,0))
            a = vt12 - cosw * (vt02 + vt13) - vt30 * cosw**2
            b = sinw * (2 * vt30 * cosw + vt02 + vt13)
            c = -vt30 * sinw**2
            desc = b**2 - 4*a*c
            index = a != 0
            nump = np.where(index, -b + sqrt(desc), -c)
            denom= np.where(index, 2*a, b)
            j = np.arctan2(nump,denom)/w
            result.append(j)
        result = np.array(result)
        return result.reshape(ll.shape)

    def invtransform_v(self, pts):
        u = pts[0].flatten()[np.newaxis]
        v = pts[1].flatten()[np.newaxis]
        a = self.ctrlpts_v[..., 0, np.newaxis]
        b = self.ctrlpts_v[..., 1, np.newaxis]
        c = self.ctrlpts_v[..., 2, np.newaxis]
        d = self.ctrlpts_v[..., 3, np.newaxis]
        f = slerp(a,b,u)
        g = slerp(d,c,u)
        h = slerp(b,c,v)
        k = slerp(a,d,v)
        inv = np.cross(np.cross(f, g, axis=0),
                       np.cross(h, k, axis=0), axis=0)
        result = UnitVector.invtransform_v(inv)
        return result.reshape(pts.shape)

class Crider(UVMapProjection):

    def transform(self, lon, lat):
        vtestpt = UnitVector.transform(lon, lat)
        ctrlpts_v = self.ctrlpts_v
        lon + 0#will TypeError if lon is not a number
        result = []
        for p in [(0, 1, 2, 3),(1, 2, 3, 0)]:
            #FIXME can calculate a lot of this stuff beforehand
            v0 = ctrlpts_v[..., p[0]]
            v1 = ctrlpts_v[..., p[1]]
            v2 = ctrlpts_v[..., p[2]]
            v3 = ctrlpts_v[..., p[3]]
            vt01 = vtestpt @ np.cross(v0, v1)
            vt12 = vtestpt @ np.cross(v1, v2)
            vt23 = vtestpt @ np.cross(v2, v3)
            vt30 = vtestpt @ np.cross(v3, v0)
            vt02 = vtestpt @ np.cross(v0, v2)
            vt13 = vtestpt @ np.cross(v1, v3)
            cosw01 = v0 @ v1
            cosw23 = v2 @ v3
            w01 = np.arccos(cosw01)
            w23 = np.arccos(cosw23)
            sinw01 = sqrt(1 - cosw01**2)
            sinw23 = sqrt(1 - cosw23**2)
            if np.isclose(w01, w23):
                w = (w01 + w23) / 2
                sinw = np.sin(w)
                cosw = np.cos(w)
                j = self._b_eq(w, sinw, cosw, vt01, vt02,
                               vt12, vt13, vt23, vt30,
                               v0, v1, v2, v3)
            else:
                j = self._b_neq(w01, w23, vt12, vt30, vt02, vt13,
                                cosw01, sinw01, cosw23, sinw23)
            result.append(j)
        return result

    def _b_neq(self, w01, w23, vt12, vt30, vt02, vt13,
               cosw01, sinw01, cosw23, sinw23):
        rx2 = [vt12 - vt30 * (cosw01 * cosw23 + sinw01 * sinw23)
                   - cosw01 * vt02 - cosw23 * vt13,
               vt30 * (sinw23 * cosw01 - sinw01 * cosw23 )
                   + sinw23 * vt13 - sinw01 * vt02,
                   + cosw01 * vt02 + cosw23 * vt13 -vt12
                   + vt30 * (cosw01 * cosw23 - sinw01 * sinw23),
               vt30 * (sinw01 * cosw23 + sinw23 * cosw01 )
                   + sinw01 * vt02 + sinw23 * vt13]
        c1 = sqrt(rx2[0]**2 +rx2[1]**2)
        c2 = sqrt(rx2[2]**2 +rx2[3]**2)
        d1 = np.arctan2(rx2[1], rx2[0])
        d2 = np.arctan2(rx2[3], rx2[2])
        wm = w01 - w23
        wp = w01 + w23
        #theta01 = np.arccos(v0 @ poip)
        #theta23 = np.arccos(v3 @ poip)
        #slerp(v0, v1, -thetap/w01) = slerp(v3, v2, -thetap2/w23) = poip
        #lims = [theta01/w01, theta23/w23
        #lim = 2
        def objective(t):
            #FIXME need a limit here to prevent results running away
            #if t < -lim or t > lim:
            #    return t**2, 2*t
            z = c1*np.cos(wm*t - d1) + c2*np.cos(wp*t - d2)
            dz = -c1*wm*np.sin(wm*t - d1) - c2*wp*np.sin(wp*t - d2)
            return z, dz

        res = root_scalar(objective, fprime=True, method='newton', x0=0.5)
        return res.root

    def _b_eq(self, w, sinw, cosw, vt01, vt02, vt12, vt13, vt23, vt30,
              v0,v1,v2,v3):
        a = vt12 - cosw * (vt02 + vt13) - vt30 * cosw**2
        b = sinw * (2 * vt30 * cosw + vt02 + vt13)
        c = -vt30 * sinw**2
        if a == 0:
            num = -c
            denom = b
            j = np.arctan2(num,denom)/w
            return j
        else:
            desc = b**2 - 4*a*c
            nump = -b + sqrt(desc)
            denom= 2*a
            jp = np.arctan2(nump,denom)/w
            return jp

    def invtransform(self, u, v):
        a = self.ctrlpts_v[..., 0]
        b = self.ctrlpts_v[..., 1]
        c = self.ctrlpts_v[..., 2]
        d = self.ctrlpts_v[..., 3]
        f = slerp(a,b,u)
        g = slerp(d,c,u)
        h = slerp(b,c,v)
        k = slerp(a,d,v)
        inv = np.cross(np.cross(f, g), np.cross(h, k))
        return UnitVector.invtransform_v(inv)


class SnyderEA4(CtrlPtsProjection):
    def __init__(self, ctrlpts, tgtpts=TGTPTS4):
        """Parameters:
        ctrlpts: 2xn Numpy array, latitude and longitude of each control point
        """
        if ctrlpts.shape[1] != tgtpts.shape[1]:
            raise ValueError('ctrlpts and tgtpts have different lengths')
        nctrlpts = ctrlpts.shape[1]
        self.nctrlpts = nctrlpts
        self.tgtpts = tgtpts
        super().__init__(ctrlpts)
        center = self.center
        bcenter = tgtpts.mean(axis=1)
        self.bcenter = bcenter
        self.btargets = [np.concatenate([bcenter[:, np.newaxis],
                                         np.roll(TGTPTS4, -i, axis=1)[:, :2]],
                                        axis=1) for i in range(nctrlpts)]
        subproj = []
        bprojs = []
        arr = np.arange(nctrlpts)
        for i in range(nctrlpts):
            index = np.roll(arr, -i)[:2]
            cp = np.concatenate([center[:,np.newaxis],
                                 ctrlpts[:, index]], axis=1)
            pj = SnyderEA(cp)
            subproj.append(pj)
            bprojs.append(Barycentric(self.btargets[i]))
        self.subproj = subproj
        self.bprojs = bprojs

        #for segment
        bc1 = np.concatenate([bcenter, [1]], axis=0)
        tgt1 = np.concatenate([tgtpts, np.ones((1,tgtpts.shape[1]))], axis=0)
        bcxtgt = -np.cross(tgt1, bc1, axis=0)
        self.bcxtgt = bcxtgt

    def transform(self, lon, lat):
        subproj = self.subproj
        bprojs = self.bprojs
        i = self.lune(lon, lat)
        pj = subproj[i]#FIXME right offset?
        bp = bprojs[i]#FIXME same
        betap = pj.transform(lon, lat)
        uvp = bp.transform(*betap)
        return uvp

    def segment(self, u, v):

        bcxtgt = self.bcxtgt
        try:
            fill = np.ones(u.shape)
        except AttributeError:
            fill = 1
        uv1 = np.stack([u,v,fill], axis=0)
        #print(bcxtgt)
        #print(uv1)
        sk = bcxtgt.T @ uv1
        sg = sk >= 0
        ind = sg & ~np.roll(sg, shift=-1, axis=0)
        result = np.argmax(ind, axis=0)
        return result#.reshape(u.shape)

    def invtransform(self, u, v):
        u + 0
        i = self.segment(u, v)
        pj = self.subproj[i]#FIXME
        bp = self.bprojs[i]
        bary = bp.invtransform(u, v)
        return pj.invtransform(*bary)


class NSlerpQuad(UVMapProjection, KProjection):

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

    def _angles_interp(self, x, y):
        """Interpolates the angle factors separately so that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        pow= self.pow
        angles = self.sides
        ax = angles[0]
        bx = angles[2]
        ay = angles[3]
        by = angles[1]
        result1 = (ax*(1-y)**pow + bx*(y)**pow)/((1-y)**pow + (y)**pow)
        result2 = (ay*(1-x)**pow + by*(x)**pow)/((1-x)**pow + (x)**pow)
        return result1, result2

    def invtransform_v(self, v):
        """
        Naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        anglex, angley = self._angles_interp(x, y)
        sx = np.sin((x)*anglex)
        sy = np.sin((y)*angley)
        scx = np.sin((1-x)*anglex)
        scy = np.sin((1-y)*angley)
        a = scx * scy
        b = sx * scy
        c = sx * sy
        d = scx * sy
        mat = (np.stack([a, b, c, d], axis=-1) /
            (np.sin(anglex)* np.sin(angley))[..., np.newaxis] )
        result = (mat.dot(self.ctrlpts_v.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(x, y, result)
        return UnitVector.invtransform_v(result)

class NSlerpQuad2(UVMapProjection, KProjection):

    def __init__(self, ctrlpts, k=1, exact = True, pow=1, eps=0):
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

    def _angles_interp(self, x, y):
        """Interpolates the angle factor together that it's equal to the
        angle between pts 1 and 2 when y=-1, etc.
        """
        pow = self.pow
        angles = self.sides
        if self.eq:
            return self.avangle
        a = ((1-x)*(1-y)*x)**pow
        b = ((1-y)*x*y)**pow
        c = ((1-x)*x*y)**pow
        d = ((1-x)*(1-y)*y)**pow
        numer = a*angles[0] + b*angles[1] + c*angles[2] + d*angles[3]
        denom = a + b + c + d
        return numer/denom

    def invtransform_v(self, v):
        """
        Variant naive slerp on a spherical quadrilateral.
        """
        x = v[0]
        y = v[1]
        angle = self._angles_interp(x, y)[..., np.newaxis]
        a = (1-x)*(1-y)
        b = x*(1-y)
        c = x*y
        d = (1-x)*y
        mat = (np.sin(np.stack([a, b, c, d], axis=-1)*angle) /
               np.sin(angle))
        result = (mat.dot(self.ctrlpts_v.T)).T
        result = self.extend(result)
        result = self._fix_corners_inv(x, y, result)
        return UnitVector.invtransform_v(result)

class EllipticalQuad(UVMapProjection, KProjection):
    """An extension of the elliptical map.
    """

    def __init__(self, ctrlpts, k=1, exact=True, eps=1E-6):
        """Parameters:
        ctrlpts: 2x4 Numpy array, latitude and longitude of each control point
        """
        super().__init__(ctrlpts)
        self.k = k
        self.exact = exact
        sidelength = self.sides
        assert abs(sidelength[0] - sidelength[2]) < eps
        assert abs(sidelength[1] - sidelength[3]) < eps
        vertangles = (np.roll(self.baz, -1) - self.faz) % 360
        assert abs((vertangles - vertangles.mean()).sum()) < eps
        ctrlpts_v = self.ctrlpts_v
        center_v = self.center_v
        midpoint_x = ctrlpts_v[:, 1] + ctrlpts_v[:, 2]
        midpoint_y = ctrlpts_v[:, 0] + ctrlpts_v[:, 1]
        m2 = np.cross(center_v, midpoint_y)
        m3 = np.cross(center_v, midpoint_x)
        mat = np.array([m2/np.linalg.norm(m2),
                        m3/np.linalg.norm(m3),
                        center_v]).T
        self.mat = mat
        self.invmat = np.linalg.inv(mat)
        self.rotctrlpts_v = self.invmat @ ctrlpts_v

    def transform(self, *args, **kwargs):
        return NotImplemented

    def invtransform_v(self, uv):
        #FIXME needs rotations
        rot_base = self.rotctrlpts_v
        a = rot_base[0,2]
        b = rot_base[1,2]
        c = rot_base[2,2]
        x = uv[0]*2 - 1
        y = uv[1]*2 - 1
        axt = (1 - a**2*x**2)
        byt = (1 - b**2*y**2)
        at = (1-a**2)
        bt = (1-b**2)
        u = a * x * sqrt(byt/bt)
        v = b * y * sqrt(axt/at)
        w = c * sqrt(axt*byt/(at*bt))
        result = np.stack([u,v,w], axis=0)
        result = self.mat @ result
        #print(v, result)
        result = self.extend(result)
        result = self._fix_corners_inv(uv[0], uv[1], result)
        return UnitVector.invtransform_v(result)