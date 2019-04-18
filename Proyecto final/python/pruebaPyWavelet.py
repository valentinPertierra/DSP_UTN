# -*- coding: utf-8 -*-
#Documentacion del modulo pywt: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html


from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom
import numpy as np
import pywt


t = np.linspace(-1, 1, 200*3, endpoint=False)
#sig  = signal.gausspulse(t - 0.4, fc=5)
sig  = np.cos(2*np.pi*7*t)
sig[0:118] = 0
sig[160:600] = 0
sig[500] = 10


#sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
widths = np.arange(1, 150)

cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus1')

plt.figure()
plt.imshow(cwtmatr, extent=[-1, 1, 1, 150],  cmap='seismic', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
plt.colorbar()

fig, ax = plt.subplots(figsize=(10,10))

ax.plot(t,sig)
ax.plot(t,cwtmatr[2,:])


ax.annotate('Aca va el texto',
            xy=(0, 0), xycoords='data',
            xytext=(50, 100), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')

ax.annotate('Aca',
            xy=(0, 5), xycoords='data',
            xytext=(50, 100), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='right', verticalalignment='bottom')

#plt.plot(t,cwtmatr[4,:])
#plt.plot(t,cwtmatr[8,:])
#plt.plot(t,cwtmatr[16,:])
plt.grid(True)
plt.show()




