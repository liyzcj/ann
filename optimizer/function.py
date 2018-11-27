# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 02:08:53 2018

@author: liyz
"""
import numpy as np


def sphere(x):
    
    y = x[0] ** 2 + x[1] ** 2
    
    return y

def rosenbrock(x):
    
    y = 100 * ((x[1] - x[0] ** 2)) ** 2 + (x[0] -1) ** 2
    
    return y

def rastrigin(x):
    
    d = 2
    y = 10 * d;
    for i in range(d):
        y += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
        
    return y

def griewank(x):
    
    d = 2
    part1 = 0;
    for i in range(d):
        part1 += x[i] ** 2
    part2 = 1
    for i in range(d):
        part2 *= np.cos(x[i] / np.sqrt(i+1))
        
    y = 1 + part1 / 4000 - part2
    
    return y