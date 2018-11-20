# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:58:23 2018

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
N = np.arange(10,5010,10)
K = 4

realizaciones = 200

# Parametros del ruido normal
u = 0   # Media
v = 2   # Varianza

resultados = []
tus_resultados = []

for Ni in N:
    
    # Determino el largo de los bloques a promediar
    L = Ni//K
    
    # Genero matriz con señales aleatoreas de ruido normal
    x = np.sqrt(v)*np.random.randn(Ni,realizaciones)+u
    
    promPSD = np.zeros(shape=(L,realizaciones))
    
    # Promedio los PSD de los K bloques de la señal
    for Ki in range(K):
        espectro = fft(x[Ki*L:(Ki+1)*L,:],axis=0)*(1/L)
        PSD = np.abs(espectro)**2
        
        promPSD = promPSD+PSD/K
    
    # Varianza del estimador
    varPSD = np.var(promPSD,axis=1)*(L**2)
    meanVarPSD = np.mean(varPSD)
    
    # Valor esperado de la PSD
    EPSD = np.mean(promPSD,axis=1)
    
    # Calculo el sesgo
    sesgo = v-np.sum(EPSD)
    
    resultados.append([sesgo,meanVarPSD])
    #print("N: "+str(Ni)+", Sesgo: "+str(sesgo)+", Varianza:"+str(meanVarPSD))

tus_resultados = [resultados[0],resultados[5-2],resultados[10-1],resultados[25-1],resultados[50-1],resultados[100-1],resultados[500-1]]

grafResultados = np.array(resultados)
N = np.arange(10,5010,10)

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.title("Módulo del sesgo")
plt.plot(N,np.abs(grafResultados[:,0]))          
plt.xlabel('N [muestras]')
plt.grid(True)

plt.subplot(212)
plt.title("Varianza")
plt.plot(N,grafResultados[:,1])
plt.axis([-10,N[-1]+10,0,1.5*v**2])
plt.xlabel('N [muestras]')
plt.grid(True)



