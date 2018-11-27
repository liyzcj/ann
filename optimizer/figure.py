# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:14:21 2018

@author: liyz
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import *


def plotfunc(func,x):
    
    x = np.linspace(-x, x)
    y = x
    xx,yy = np.meshgrid(x,y)
    zz = func((xx,yy))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(xx,yy,zz,
                    rstride=1, 
                    cstride=1, 
                    cmap='rainbow'
                    )
    
    ax.contourf(xx,yy,zz, zdir='z', offset=0, cmap='rainbow')
    plt.show()
    
if __name__ == '__main__':
    
#    plotfunc(sphere, 5.12)
#    plotfunc(rosenbrock, 2.048)
    plotfunc(rastrigin, 5.12)
#    plotfunc(griewank, 600)