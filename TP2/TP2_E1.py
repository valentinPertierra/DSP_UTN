# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:49:22 2018

@author: Valentin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML

from funciones import *
from scipy import signal
from ventanas import *

N  = 1000 # muestras
fs = 1000 # Hz


#k = np.arange(0,N)
t = np.linspace(0,(N-1)/fs,N)
k = t
wBartlett = bartlett(N)
wHann = hann(N)
wBlackman = blackman(N)
wFlatTop = flatTop(N)

f,eBartlett = analizadorEspectro(wBartlett)
f,eHann = analizadorEspectro(wHann)
f,eBlackman = analizadorEspectro(wBlackman)
f,eFlatTop = analizadorEspectro(wFlatTop)
'''
f,eBartlett = analizadorEspectro2(wBartlett,t)
f,eHann = analizadorEspectro2(wHann,t)
f,eBlackman = analizadorEspectro2(wBlackman,t)
f,eFlatTop = analizadorEspectro2(wFlatTop,t)
'''
lps = np.finfo('float').eps
print(lps)
# Grafico las ventanas con sus espectros
plt.figure(figsize=(16,16))
plt.subplot(421)
plt.title("Ventana Bartlett")
plt.plot(k, wBartlett)           
plt.xlabel('k muestras')
plt.ylabel('w(k)')
plt.grid(True)

plt.subplot(422)
plt.title("Modulo del espectro de la ventana Bartlett")
plt.plot(f, 20*np.log10(np.absolute(eBartlett)+lps))           
plt.xlabel('Frecuencia normalizada $\Omega $ [$\Pi $rad/muestra]')
plt.ylabel('$ |W(\Omega)| [dB] $')
#plt.axis([0,200,0,np.amax(np.absolute(DFTsenoidal))*1.1])
plt.grid(True)

plt.subplot(423)
plt.title("Ventana Hann")
plt.plot(k, wHann)           
plt.xlabel('k')
plt.ylabel('w(k)')
plt.grid(True)

plt.subplot(424)
plt.title("Modulo del espectro de la ventana Hann")
plt.plot(f, 20*np.log10(np.absolute(eHann)+lps))           
plt.xlabel('$\Omega $')
plt.ylabel('$ |W(\Omega)| [dB] $')
#plt.axis([0,200,0,np.amax(np.absolute(DFTsenoidal))*1.1])
plt.grid(True)

plt.subplot(425)
plt.title("Ventana Blackman")
plt.plot(k, wBlackman)           
plt.xlabel('k')
plt.ylabel('w(k)')
plt.grid(True)

plt.subplot(426)
plt.title("Modulo del espectro de la ventana Blackman")
plt.plot(f, 20*np.log10(np.absolute(eBlackman)+lps))           
plt.xlabel('$\Omega $')
plt.ylabel('$ |W(\Omega)| [dB] $')
#plt.axis([0,200,0,np.amax(np.absolute(DFTsenoidal))*1.1])
plt.grid(True)

plt.subplot(427)
plt.title("Ventana FlatTop")
plt.plot(k, wFlatTop)           
plt.xlabel('k')
plt.ylabel('w(k)')
plt.grid(True)

plt.subplot(428)
plt.title("Modulo del espectro de la ventana FlatTop")
plt.plot(f, 20*np.log10(np.absolute(eFlatTop)+lps))           
plt.xlabel('$\Omega $')
plt.ylabel('$ |W(\Omega)| [dB] $')
#plt.axis([0,200,0,np.amax(np.absolute(DFTsenoidal))*1.1])
plt.grid(True)

plt.show()





