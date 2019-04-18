# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:07:14 2019

@author: Valentin
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-1, 1, 200*3, endpoint=False)
sig  = signal.gausspulse(t - 0.4, fc=2)
#sig = np.zeros(len(t))
sig[20] = 10

widths = np.arange(1, 31*3)
cwtmatr = signal.cwt(sig, signal.ricker, widths)

plt.imshow(cwtmatr, extent=[-1, 1,1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

# Grafico la señal en el timepo
plt.figure()
plt.plot(t,sig)
plt.plot(t,cwtmatr[5,:])
plt.grid(True)
plt.show()


