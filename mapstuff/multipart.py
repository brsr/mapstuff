# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:09:25 2023

@author: brsto
"""
import numpy as np
import matplotlib.pyplot as plt
import pyproj

from .transformations import UnitVector, Transformation, Barycentric
from .helper import sqrt, shoelace
#from mapstuff.tri import SnyderEA
_geod = pyproj.Geod(a=6371, f=0)

#%% icosahedron
lat = np.concatenate([[90],
                      np.arctan(1/2)*180/np.pi*np.tile([1.0, -1.0], 5),
                      [-90]])
lon = np.concatenate([[0],
                      np.linspace(-180, 180, num=10, endpoint=False),
                      [0]])

triangles = np.array([[0,1,3],
                      [0,3,5],
                      [0,5,7],
                      [0,7,9],
                      [0,9,1],

                      [1,2,3],
                      [3,4,5],
                      [5,6,7],
                      [7,8,9],
                      [9,10,1],

                      [4,3,2],
                      [6,5,4],
                      [8,7,6],
                      [10,9,8],
                      [2,1,10],

                      [11,4,2],
                      [11,6,4],
                      [11,8,6],
                      [11,10,8],
                      [11,2,10],
                      ])

tlat = lat[triangles]
tlon = lon[triangles]

# for a, b in zip(tlon, tlat):
#     print(geod.polygon_area_perimeter(a, b, radians=True))

# plt.scatter(lon, lat)
# plt.plot(tlon.T, tlat.T)

# v = UnitVector.transform(tlon, tlat, scale=1)
# print(np.linalg.det(v.transpose(1,0,2)))

#vx = UnitVector.transform(lon, lat, scale=1)

# plt.plot(vx[0,triangles].T, vx[1,triangles].T)
# #subdivision
# tri = triangles[0]
# vtri = vx[:,tri]
# center = vtri.sum(axis=1)
# center = normalize(center)
# midpoints = np.roll(vtri, -1, axis=1) + np.roll(vtri, 1, axis=1)
# midpoints= normalize(midpoints)

# subtri = np.array([[center, vtri[:,0], midpoints[:,2]],
#                    [center, midpoints[:,2], vtri[:,1]],
#                    [center, vtri[:,1], midpoints[:,0]],
#                    [center, midpoints[:,0], vtri[:,2]],
#                    [center, vtri[:,2], midpoints[:,1]],
#                    [center, midpoints[:,1], vtri[:,0]]]).T

# plt.plot(vtri[0, cycle], vtri[1, cycle])
# plt.plot(subtri[0,cycle], subtri[1,cycle])
#%% icosahedron target
cont = 1j*sqrt(3)/2
tgttriangles = np.array([[0.5+cont, 0, 1],
                        [1.5+cont, 1, 2],
                        [2.5+cont, 2, 3],
                        [3.5+cont, 3, 4],
                        [4.5+cont, 4, 5],

                        [0, 0.5-cont, 1],
                        [1, 1.5-cont, 2],
                        [2, 2.5-cont, 3],
                        [3, 3.5-cont, 4],
                        [4, 4.5-cont, 5],

                        [1.5-cont, 1, 0.5-cont],
                        [2.5-cont, 2, 1.5-cont],
                        [3.5-cont, 3, 2.5-cont],
                        [4.5-cont, 4, 3.5-cont],
                        [5.5-cont, 5, 4.5-cont],

                        [1-2*cont, 1.5-cont, 0.5-cont],
                        [2-2*cont, 2.5-cont, 1.5-cont],
                        [3-2*cont, 3.5-cont, 2.5-cont],
                        [4-2*cont, 4.5-cont, 3.5-cont],
                        [5-2*cont, 5.5-cont, 4.5-cont]])

tgttriangles += 2*cont
tgttriangles *= 2*sqrt(np.pi/5/sqrt(3))*_geod.a
#plt.plot(tgttriangles.real[:, cycle].T, tgttriangles.imag[:, cycle].T)

# tgttri = tgttriangles[0]
# center = tgttri.mean()

# midpoints = (np.roll(tgttri, -1) + np.roll(tgttri, 1))/2

# subtgttri = np.array([[center, tgttri[0], midpoints[2]],
#                    [center, midpoints[2], tgttri[1]],
#                    [center, tgttri[1], midpoints[0]],
#                    [center, midpoints[0], tgttri[2]],
#                    [center, tgttri[2], midpoints[1]],
#                    [center, midpoints[1], tgttri[0]]]).T

# plt.plot(tgttri.real[cycle], tgttri.imag[cycle])
# plt.plot(subtgttri.real[cycle], subtgttri.imag[cycle])

class Icosahedron():
    def __init__(self):
        #FIXME probably wrong index order
        self.sph_tri = np.array([tlon,
                                 tlat]).transpose(1, 0, 2)
        self.vec_tri = UnitVector.transform(tlon, tlat).transpose(1,0,2)
        self.tgt_tri = np.array([tgttriangles.real,
                                 tgttriangles.imag]).transpose(1, 0, 2)

    def subdivide_sph(self, roll=0):
        """
        0: center is first
        1: midpoint is first
        2: vertex is first
        """
        result = []
        for vtri, tri in zip(self.vec_tri, self.sph_tri):
            #print(tri)
            vcenter = vtri.sum(axis=1)
            #vcenter = normalize(vcenter)
            center = UnitVector.invtransform_v(vcenter)#, scale=1)
            #print(center)
            vmidpoints = np.roll(vtri, -1, axis=1) + np.roll(vtri, 1, axis=1)
            #vmidpoints = normalize(vmidpoints)
            midpoints = UnitVector.invtransform_v(vmidpoints)#, scale=1)
            #print(midpoints)

            subtri = [np.roll([center, tri[:,0], midpoints[:,2]], roll, axis=0),
                      np.roll([center, midpoints[:,2], tri[:,1]], -roll, axis=0),
                      np.roll([center, tri[:,1], midpoints[:,0]], roll, axis=0),
                      np.roll([center, midpoints[:,0], tri[:,2]], -roll, axis=0),
                      np.roll([center, tri[:,2], midpoints[:,1]], roll, axis=0),
                      np.roll([center, midpoints[:,1], tri[:,0]], -roll, axis=0)]
            result.extend(subtri)
        return np.array(result).transpose(0,2,1)

    def subdivide_tgt(self, roll=0):
        complex_tgt_tri = self.tgt_tri[:,0] + 1j*self.tgt_tri[:,1]
        result = []
        for tgttri in complex_tgt_tri:
            center = tgttri.mean()

            midpoints = (np.roll(tgttri, -1) + np.roll(tgttri, 1))/2

            subtgttri = [np.roll([center, tgttri[0], midpoints[2]], roll),
                         np.roll([center, midpoints[2], tgttri[1]], -roll),
                         np.roll([center, tgttri[1], midpoints[0]], roll),
                         np.roll([center, midpoints[0], tgttri[2]], -roll),
                         np.roll([center, tgttri[2], midpoints[1]], roll),
                         np.roll([center, midpoints[1], tgttri[0]], -roll)]
            result.extend(subtgttri)
        result = np.array(result)
        return np.array([result.real, result.imag]).transpose(1,0,2)
    
icosa = Icosahedron()
# cycle = [-1, 0, 1, 2]

# subsph = icosa.subdivide_sph()
# plt.plot(icosa.sph_tri[..., 0, cycle].T, icosa.sph_tri[..., 1, cycle].T)
# plt.plot(subsph[..., 0, cycle].T, subsph[..., 1, cycle].T)

# for a, b in subsph:
#     print(_geod.polygon_area_perimeter(a, b))
#%%
# subsph = icosa.subdivide_sph(1)
# plt.plot(icosa.sph_tri[..., 0, cycle].T, icosa.sph_tri[..., 1, cycle].T)
# plt.plot(subsph[..., 0, cycle].T, subsph[..., 1, cycle].T)

# for a, b in subsph:
#     print(geod.polygon_area_perimeter(a, b, radians=True))
# #%%
# subsph = icosa.subdivide_sph(2)
# plt.plot(icosa.sph_tri[..., 0, cycle].T, icosa.sph_tri[..., 1, cycle].T)
# plt.plot(subsph[..., 0, cycle].T, subsph[..., 1, cycle].T)

# for a, b in subsph:
#     print(geod.polygon_area_perimeter(a, b, radians=True))


# #%%
# subtgt = icosa.subdivide_tgt()    
# for t in icosa.tgt_tri:
#     print(shoelace(t))

# for t in subtgt:
#     print(shoelace(t))

# plt.plot(icosa.tgt_tri[..., 0, cycle].T, icosa.tgt_tri[..., 1, cycle].T)
# plt.plot(subtgt[..., 0, cycle].T, subtgt[..., 1, cycle].T)

#%%
class MultipartProjection(Transformation):
    def __init__(self, projl, transl):
        self.projl = projl
        self.transl = transl
        ctrlpts_l = [x.ctrlpts for x in projl]
        tgtpts_l = [x.tgtpts for x in transl]
        ctrlpts_v_l = [UnitVector.transform(*x) for x in ctrlpts_l]
        segl = zip(projl, transl, ctrlpts_v_l, tgtpts_l)
        segments = []
        for proj, trans, ctrlpts_v, tgtpts in segl:
            invinv = np.linalg.pinv(np.concatenate([tgtpts,
                                                    np.ones((1,3))],axis=0))
            segments.append({'proj': proj,
                            'trans': trans,
                            'ctrlpts_v': ctrlpts_v,
                            'invmat': np.linalg.pinv(ctrlpts_v),
                            'tgtpts': tgtpts,
                            'invinvmat': invinv})
        self.segments = segments


    def transform(self, lon, lat):
        lon + 0
        ptv = UnitVector.transform(lon, lat)
        s = []
        for segment in self.segments:
            invmat = segment['invmat']
            #print(invmat)
            res = invmat @ ptv
            s.append(res.min())
            #print(ptv, res)
            if np.all(res >= 0):
                break
        i = np.argmax(s)
        segment = self.segments[i]
        pj = segment['proj']
        result = pj.transform(lon, lat)
        trans = segment['trans']
        tresult = trans.transform(*result)
        return tresult

    def invtransform(self, x, y, z=None, **kwargs):
        x + 0
        bary = np.stack([x, y, np.ones(x.shape)])
        s = []
        for segment in self.segments:
            invmat = segment['invinvmat']
            #print(invmat)
            res = invmat @ bary
            s.append(res.min())
            #print(ptv, res)
            if np.all(res >= 0):
                break
        i = np.argmax(s)
        segment = self.segments[i]
        trans = segment['trans']
        tresult = trans.invtransform(x, y)
        pj = segment['proj']
        result = pj.invtransform(*tresult)
        return result

class IcosahedralProjection(MultipartProjection):
    def __init__(self, proj, trans=Barycentric,
                 ctrltriangles=icosa.sph_tri,
                 tgttriangles=icosa.tgt_tri):
        projl = []
        for ct in ctrltriangles:
            projl.append(proj(ct))
        transl = []
        for tt in tgttriangles:
            transl.append(trans(tt))
        super().__init__(projl, transl)

class SubdividedIcosahedralProjection(MultipartProjection):
    def __init__(self, proj, trans=Barycentric, first=0, icoobj = icosa):
        projl = []
        for ct in icosa.subdivide_sph(first):
            projl.append(proj(ct))
        transl = []
        for tt in icosa.subdivide_tgt(first):
            transl.append(trans(tt))
        super().__init__(projl, transl)
        
class MultipartProjectionNoPost(Transformation):
    def __init__(self, projl):
        self.projl = projl
        ctrlpts_l = [x.ctrlpts for x in projl]
        tgtpts_l = [x.tgtpts for x in projl]
        #print(tgtpts_l)
        ctrlpts_v_l = [UnitVector.transform(*x) for x in ctrlpts_l]
        segl = zip(projl, ctrlpts_v_l, tgtpts_l)
        segments = []
        for proj, ctrlpts_v, tgtpts in segl:
            invinv = np.linalg.pinv(np.concatenate([tgtpts,
                                                    np.ones((1,3))],axis=0))
            segments.append({'proj': proj,
                            'ctrlpts_v': ctrlpts_v,
                            'invmat': np.linalg.pinv(ctrlpts_v),
                            'tgtpts': tgtpts,
                            'invinvmat': invinv})
        self.segments = segments


    def transform(self, lon, lat):
        lon + 0
        ptv = UnitVector.transform(lon, lat)
        s = []
        for segment in self.segments:
            invmat = segment['invmat']
            #print(invmat)
            res = invmat @ ptv
            s.append(res.min())
            #print(ptv, res)
            if np.all(res >= 0):
                break
        i = np.argmax(s)
        segment = self.segments[i]
        pj = segment['proj']
        result = pj.transform(lon, lat)
        return result

    def invtransform(self, x, y, z=None, **kwargs):
        x + 0
        bary = np.stack([x, y, np.ones(x.shape)])
        s = []
        for segment in self.segments:
            invmat = segment['invinvmat']
            #print(invmat)
            res = invmat @ bary
            s.append(res.min())
            #print(ptv, res)
            if np.all(res >= 0):
                break
        i = np.argmax(s)
        segment = self.segments[i]
        pj = segment['proj']
        result = pj.invtransform(x, y)
        return result
        
class IcosahedralProjectionNoPost(MultipartProjectionNoPost):
    def __init__(self, proj, trans=Barycentric,
                 ctrltriangles=icosa.sph_tri,
                 tgttriangles=icosa.tgt_tri):
        projl = []
        for ct, tt in zip(ctrltriangles, tgttriangles):
            projl.append(proj(ct, tt.copy()))
        super().__init__(projl)
