#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:24:35 2021

@author: brsr
"""

import numpy as np

n = 3
vi = np.random.normal(size=(3,n))
vi /= np.linalg.norm(vi, axis=0, keepdims=True)
#exact solution if n=3
xx = np.cross(vi, np.roll(vi, 1, axis=1), axis=0).sum(axis=1)
xx /= np.linalg.norm(xx)

y = np.ones(n)
res = np.linalg.lstsq(vi.T, y, rcond=None)
x = res[0].copy()
#x /= np.linalg.norm(x)

m = vi @ vi.T
y2 = vi.sum(axis=1)

minv = np.empty(dtype=float, shape=(3,3))

minv[0,0] = m[1,1]*m[2,2] - m[2,1]**2
minv[1,1] = m[2,2]*m[0,0] - m[0,2]**2
minv[2,2] = m[0,0]*m[1,1] - m[1,0]**2
minv[0,1] = m[0,2]*m[1,2] - m[0,1]*m[2,2] 
minv[0,2] = m[0,1]*m[1,2] - m[0,2]*m[1,1] 
minv[1,2] = m[0,1]*m[0,2] - m[0,0]*m[1,2]
minv[1,0] = minv[0,1] 
minv[2,0] = minv[0,2] 
minv[2,1] = minv[1,2] 

x2 = minv @ y2
print(y2)
print(x)
print(x2)
print(x2/x)
#(a | b | c
#b | d | e
#c | e | f)^(-1) (matrix inverse)
#= k * 
#(e^2 - d f | b f - c e | c d - b e
#b f - c e | c^2 - a f | a e - b c
#c d - b e | a e - b c | b^2 - a d)
