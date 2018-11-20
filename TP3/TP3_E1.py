# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:35:33 2018

@author: Valentin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML

from funciones import *
from scipy.fftpack import fft

fs = 1000 # Hz

# Simular para los siguientes tamaños de señal
#N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=np.float)
#N = [10, 50, 100, 250, 500, 1000, 5000]
N = np.arange(10,5000,10)
realizaciones = 200

# Parametros del ruido normal
u = 0   # Media
v = 2   # Varianza

tus_resultados = []

for Ni in N:
    
    # Genero matriz con señales aleatoreas de ruido normal
    x = np.sqrt(v)*np.random.randn(Ni,realizaciones)+u
    
    # Obtengo el espectro
    espectro = fft(x,axis=0)*(1/Ni)
    
    # Obtengo la densidad espectral de potencia PSD
    PSD = np.abs(espectro)**2
    
    # Varianza del periodograma
    varPSD = np.var(PSD,axis=1)*(Ni**2)
    meanVarPSD = np.mean(varPSD)
    
    # Valor esperado de la PSD
    EPSD = np.mean(PSD,axis=1)
    
    # Calculo el sesgo
    sesgo = v-np.sum(EPSD)
    
    #varianza = np.sum(EPSD)
    tus_resultados.append([sesgo,meanVarPSD]) 
    
    #print("N: "+str(Ni)+", Sesgo: "+str(sesgo)+", Varianza:"+str(meanVarPSD))

tus_resultados = np.array(tus_resultados)

plt.figure(0)
plt.title("Sesgo")
plt.plot(N,np.abs(tus_resultados[:,0]))
plt.grid()

plt.figure(1)
plt.title("Varianza")
plt.plot(N,tus_resultados[:,1])



























