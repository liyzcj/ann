import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from function import *

# ====================================
# Global parameter 
lr = 1e-2
scope = (-1, 1)
func = monkey_saddle
dfunc = d_monkey_saddle
# Initial point
starter = np.array([[-0.25,-0.25]])

# ====================================
# Contour of loss function
x = np.linspace(*scope)
xx,yy = np.meshgrid(x,x)
zz = func((xx,yy))

# ====================================
# optimizer

#--------------SGD----------------
sgddata = starter.copy()
def sgd(x,dx):
    new = x - lr*dx
    return new

#-------------Momentum------------
momdata = starter.copy()
v_mom = np.zeros(2)
mu = 0.9
def momentum(x,dx):
    global v_mom
    v_mom = mu * v_mom - lr * dx
    new = x + v_mom
    return new

# -------------Nesterov ------------
nesdata = starter.copy()
v_nes = np.zeros(2)
v_prev = np.zeros(2)

def nesterov(x, dx):
    global v_prev
    global v_nes

    v_prev = v_nes # back this up
    v_nes = mu * v_nes - lr * dx
    new = x - mu * v_prev + (1 + mu) * v_nes
    return new


# ---------- optimizers dict ------
optimizers = {'sgd','momentum','nesterov'}

optimizer_functions = {'sgd':sgd,
              'momentum':momentum,
              'nesterov':nesterov}

datas = {'sgd':sgddata,
        'momentum':momdata,
        'nesterov':nesdata}

# ====================================
# Figure
fig, ax = plt.subplots()
lines = {}
for optimizer in optimizers:
    data = datas[optimizer]
    line, = ax.plot(data[:,0], data[:,1], label=optimizer)
    scatter, = ax.plot(data[-1,0], data[-1,1], 'o', c = line.get_color())
    lines[optimizer] = [line, scatter]


# ====================================
# Animation function
def update(i):
    global datas
    # loss function contour
    ax.contour(xx, yy, zz)
    out = []
    for optimizer in optimizers:
        # set local optimizer
        data = datas[optimizer]
        optimizer_function = optimizer_functions[optimizer]
        line = lines[optimizer][0]
        scatter = lines[optimizer][1]

        # update data
        grad = dfunc(data[-1])
        new = optimizer_function(data[-1], grad)
        datas[optimizer] = np.append(data, np.array([new]), axis=0)
        line.set_data(data[:,0], data[:,1])
        scatter.set_data(data[-1,0], data[-1,1])
        out.append(line)
        out.append(scatter)
        
        plt.legend()
    return out


ani = animation.FuncAnimation(fig, update, interval=10,
                              blit=True)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()