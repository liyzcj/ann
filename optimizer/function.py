# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 02:08:53 2018

@author: liyz
"""
import numpy as np

# ----------------Function---------------------
def sphere(x):
    
    y = x[0] ** 2 + x[1] ** 2
    
    return y

def rosenbrock(x):
    
    y = 100 * ((x[1] - x[0] ** 2)) ** 2 + (x[0] -1) ** 2
    
    return y

def rastrigin(x):
    
    d = 2
    y = 10 * d
    for i in range(d):
        y += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
        
    return y

def griewank(x):
    
    d = 2
    part1 = 0
    for i in range(d):
        part1 += x[i] ** 2
    part2 = 1
    for i in range(d):
        part2 *= np.cos(x[i] / np.sqrt(i+1))
        
    y = 1 + part1 / 4000 - part2
    
    return y

def beale(x):
    x1 = x[0]
    x2 = x[1]

    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2

    y = term1 + term2 + term3
    return y

def monkey_saddle(x):

    x1 = x[0]
    x2 = x[1]

    y = x1**3 - 3 * x1 * x2**2

    return y
# ------------------Derivative---------------
def d_sphere(x):
    
    return 2*x
    
def d_beale(x):
    x1 = x[0]
    x2 = x[1]

    g1 = 2 * (1.5- x1 + x1*x2)
    g2 = 2 * (2.25 - x1 + x1*x2**2)
    g3 = 2 * (2.625 - x1 + x1 * x2**3)

    p1 = - g1 * x2
    p2 = - g2 * x2**2
    p3 = - g3 * x2**3

    dx1 = p1 + p2 + p3

    p1 = g1 * x1
    p2 = g2 * 2 * x1 * x2
    p3 = g3 * 3 * x1 * x2**2

    dx2 = p1 + p2 + p3

    return np.array([dx1, dx2])


def d_monkey_saddle(x):

    x1 = x[0]
    x2 = x[1]

    dx1 = 3 * x1**2 - 3 * x2**2
    dx2 = -6 * x1 * x2

    return np.array([dx1,dx2])