import time
from dataclasses import dataclass
import math
import numpy as np
import scipy
import scipy.optimize

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

draw_queue = queue.Queue()


class SetN:
    def __init__(self, n):
        self.n = n

@dataclass
class Context:
    n = 500
    c = 3.94
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10
    resolution = 0.1

    def __post_init__(self):
        self.initial_c = self.c


context = Context()


def fakesin(x, c = None, n = None):
    if c is None:
        c = context.c

    if n is None:
        n = context.n

    def t(x):
        return np.arctan((x*c)/(2*n))

    sign = 1 if n % 2 == 0 else -1

    product = np.ones_like(x)
    for i in range(1, n+1):
        product *= t(x - i * math.pi) * t(x + (i - 1) * math.pi)

    return sign * product


def objective(c, n = None):
    if n is None:
        n = context.n

    x = np.arange(context.xmin, context.xmax, context.resolution)

    y1 = np.sin(x)
    y2 = fakesin(x, c, n)

    error_terms = np.absolute(y1 - y2)
    error = np.sum(np.square(error_terms))
    return error



    


def optimizer():
    def callback(res):
        draw_queue.put(res)


    for n0 in range(200, 2000, 25):

        callback(SetN(n0))

        x0 = context.initial_c
        res = scipy.optimize.minimize(objective, x0, args=(n0,), callback=callback)

        callback(res)

        time.sleep(0.001)
        
    callback(None) # Poison pill

    return res



def draw_loop():

    def frames():
        nonlocal n

        while True:
            res = draw_queue.get() # Will block

            if isinstance(res, SetN):
                n = res.n
                continue

            if res is None:
                break

            yield res

    n = context.n
    c = context.c
    err = ""

    fig, ax = plt.subplots()

    xmin = context.xmin
    xmax = context.xmax
    ymin = context.ymin
    ymax = context.ymax

    x = np.arange(xmin, xmax, context.resolution)
    y1 = np.sin(x)
    y2 = fakesin(x, c, n)

    ln1, = plt.plot(x, y1)
    ln2, = plt.plot(x, y2)
    n_text_obj = ax.text(xmin+0.5, ymax-0.5, f"n = {n}", horizontalalignment='left', verticalalignment='center')
    c_text_obj = ax.text(xmax-0.5, ymax-0.5, f"c = {c}", horizontalalignment='right', verticalalignment='center')
    err_text_obj = ax.text(xmax-0.5, ymax-1.5, f"err = {err}", horizontalalignment='right', verticalalignment='center')
    done_text_obj = ax.text(xmax-0.5, ymax-2.5, "", horizontalalignment='right', verticalalignment='center')

    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return ln2, n_text_obj, c_text_obj, err_text_obj, done_text_obj

    def update(res):
        done = False
        if isinstance(res, scipy.optimize.OptimizeResult):
            done = True
            res_ = res
            res = res_.x

        c = res[0]
        y2 = fakesin(x, c)
        err = objective(c)
        ln2.set_data(x, y2)
        n_text_obj.set_text(f"n = {n}")
        c_text_obj.set_text(f"c = {c}")
        err_text_obj.set_text(f"err = {err}")

        if done:
            done_text_obj.set_text("Done")
        else:
            done_text_obj.set_text("")

        return ln2, n_text_obj, c_text_obj, err_text_obj, done_text_obj

    ani = FuncAnimation(
        fig, 
        update, 
        frames=frames,
        init_func=init,
        blit=True,
    )

    plt.show()


def draw_simple():
    x = np.arange(context.xmin, context.xmax, context.resolution)
    y1 = np.sin(x)
    y2 = fakesin(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()


optimizer_thread = threading.Thread(target=optimizer)
optimizer_thread.start()

draw_loop()
