# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:23:16 2018

@author: Valentin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML

from funciones import *
from ventanas import *
from scipy.fftpack import fft

fs = 1000 # Hz

# Simular para los siguientes tamaños de señal
#N = [10, 50, 100, 250, 500, 1000, 5000]
#N = np.arange(50,5050,50)
N =[1000]
K = 4       # Cantidad de bloques
O = 0.5       # Solapamiento

realizaciones = 200

# Parametros del ruido normal
u = 0   # Media
v = 2   # Varianza

resultados = []
tus_resultados = []

for Ni in N:
    # Determino el largo de los bloques a promediar
    # N=L+D*(K-1)        Siendo D el desplazamiento de los bloques en muentras
    # D=(1-O)*L 
    # L=N/(1+(1-O)*(K-1))   Con este valor de L vuelvo a calcular D para que sea un número entero
    L = Ni/(1+(1-O)*(K-1))
    
    D = (1-O)*L 
    D = np.round(D)
    L = Ni-D*(K-1)
    
    D = int(D)
    L = int(L)
    
    # Genero matriz con señales aleatoreas de ruido normal
    x = np.sqrt(v)*np.random.randn(Ni,realizaciones)+u
    
    promPSD = np.zeros(shape=(L,realizaciones))
    promPSDW = np.zeros(shape=(L,realizaciones))
    xw = np.zeros(shape=(L,realizaciones))
    
    # Promedio los PSD de los K bloques de la señal
    for Ki in range(K):
        
        # Sin ventanear
        xw = x[Ki*D:Ki*D+L,:]
        
        espectro = fft(xw,axis=0)*(1/L)
        
        PSD = np.abs(espectro)**2
        promPSD = promPSD+PSD/K
        
        # Aplicando una ventana al bloque
        w = np.reshape(bartlett(L),(L,1))        
        xw = x[Ki*D:Ki*D+L,:]*w
               
        espectroW = fft(xw,axis=0)*(1/L)
        
        PSDW = np.abs(espectroW)**2
        promPSDW = promPSDW+PSDW/K
    
    # Resultados sin ventaneo
    # Varianza del estimador
    varPSD = np.var(promPSD,axis=1)*(L**2)
    meanVarPSD = np.mean(varPSD)
    
    # Valor esperado de la PSD
    EPSD = np.mean(promPSD,axis=1)
    
    # Calculo el sesgo
    sesgo = v/Ni-np.mean(EPSD)
     
    # Resultados aplicando la ventana:
    # Varianza del estimador
    varPSDW = np.var(promPSDW,axis=1)*(L**2)
    meanVarPSDW = np.mean(varPSDW)
    
    # Valor esperado de la PSD
    EPSDW = np.mean(promPSDW,axis=1)
    
    # Calculo el sesgo
    sesgoW = v/Ni-np.mean(EPSDW)
    
    resultados.append([sesgo,meanVarPSD,sesgoW,meanVarPSDW])
    #print("N: "+str(Ni)+", Sesgo: "+str(sesgo)+", Varianza:"+str(meanVarPSD))


tus_resultados = [resultados[0],resultados[1],resultados[4],resultados[9],resultados[19],resultados[99]]

grafResultados = np.array(resultados)
N = np.arange(50,5050,50)

plt.figure()
plt.subplot(211)
plt.title("Módulo del sesgo")
plt.plot(N,np.abs(grafResultados[:,0]),label="sin ventana")    
plt.plot(N,np.abs(grafResultados[:,2]),label="Bartlett")       
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)

plt.subplot(212)
plt.title("Varianza")
plt.plot(N,grafResultados[:,1],label="sin ventana")
plt.plot(N,grafResultados[:,3],label="Bartlett")
plt.axis([-10,N[-1]+10,0,1.5*v**2])
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)



