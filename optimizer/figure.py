# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:14:21 2018

@author: liyz
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import *


def plotfunc(func,x,offset=0):
    
    x = np.linspace(-x, x)
    xx,yy = np.meshgrid(x,x)
    zz = func((xx,yy))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(xx,yy,zz,
                    rstride=1, 
                    cstride=1, 
                    # cmap='rainbow'
                    )
    
    ax.contourf(xx,yy,zz, zdir='z', offset=offset, cmap='rainbow')
    plt.show()

def plotcontour(func,x):

    x = np.linspace(-x, x)
    xx,yy = np.meshgrid(x,x)
    zz = func((xx,yy))

    plt.figure()
    plt.contour(xx, yy, zz)

    plt.show()
    
if __name__ == '__main__':
    
#    plotfunc(sphere, 5.12)
#    plotfunc(rosenbrock, 2.048)
    # plotfunc(rastrigin, 5.12)
#    plotfunc(griewank, 600)
    # plotcontour(beale, 20)
    plotfunc(monkey_saddle, 1, offset=-2)