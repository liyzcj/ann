import mpl_toolkits.mplot3d.axes3d as p3
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
eps = 1e-8
# Initial point
starter = np.array([[0,-0.25]])

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

# -----------Adagrad----------------
adagrad_data = starter.copy()
# grad_square_sum
gss = np.zeros(2)

def adagrad(x, dx):
    global gss
    gss += dx**2
    adjusted_dx = dx / (np.sqrt(gss) + eps)
    new = x - lr * adjusted_dx
    return new

#---------- Adadelta --------------
rho = 0.95
adadelta_data = starter.copy()
# grad_square_sum for adadelta
gssa = np.zeros(2)
# delta square sum
dss = np.zeros(2)

def adadelta(x, dx):
    global gssa
    global dss
    gssa = rho * gssa + (1-rho) * dx**2
    delta = np.sqrt(dss+eps) / np.sqrt(gssa+eps) * dx
    new = x - delta
    dss = rho * dss + (1-rho) * delta**2
    return new

#--------------RMSprop-------------
rmsprop_data = starter.copy()
#decay_rate
dr = 0.9
# grad_square_sum for RMSprop
gssr = np.zeros(2)

def rmsprop(x, dx):
    global gssr
    gssr = dr * gssr + (1-dr) * dx**2
    adjusted_dx = dx / (np.sqrt(gssr)+eps)
    new = x - lr * adjusted_dx
    return new

# -------------Adam----------------
adam_data = starter.copy()
beta1 = 0.9
beta2 = 0.999
# momentum
m = np.zeros(2)
v_adam = np.zeros(2)

def adam(x, dx):
    global m
    global v_adam
    m = beta1*m + (1-beta1)*dx
    v_adam = beta2*v_adam + (1-beta2)*(dx**2)
    new = x - lr * m / (np.sqrt(v_adam) + eps)
    return new

# ---------- optimizers dict ------
optimizers = {'sgd','momentum','nesterov','adagrad','adadelta','rmsprop','adam'}

optimizer_functions = {'sgd':sgd,
              'momentum':momentum,
              'nesterov':nesterov,
              'adagrad':adagrad,
              'adadelta':adadelta,
              'rmsprop':rmsprop,
              'adam':adam
              }

datas = {'sgd':sgddata,
        'momentum':momdata,
        'nesterov':nesdata,
        'adagrad':adagrad_data,
        'adadelta':adadelta_data,
        'rmsprop':rmsprop_data,
        'adam':adam_data
        }

# ====================================
# Figure
# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)
lines = {}
for optimizer in optimizers:
    data = datas[optimizer]
    line, = ax.plot(data[:,0], data[:,1],func(data[-1]),label=optimizer)
    scatter, = ax.plot(data[:,0], data[:,1],func(data[-1]), marker='o',c = line.get_color())
    lines[optimizer] = [line, scatter]


# ====================================
# Animation function
def update(w):
    global datas
    # loss function contour
    ax.plot_surface(xx,yy,zz,
                    rstride=1, 
                    cstride=1, 
                    cmap='Blues'
                    )
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
        zdata = np.zeros(data.shape[0])
        for i,d in enumerate(data):
            zdata[i] = func(d)
        line.set_data(data[:,0], data[:,1])
        line.set_3d_properties(zdata)
        scatter.set_data(data[-1,0],data[-1,1])
        scatter.set_3d_properties(zdata[-1])
        
        out.append(line)
        out.append(scatter)

        plt.legend()
    return out


ani = animation.FuncAnimation(fig, update,50, repeat=False, interval=10,
                              blit=True)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# import os
# import matplotlib
# ffmpegpath = os.path.abspath("path/to/ffmepg")
# matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()