# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:04:25 2018

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
#N = np.arange(100,5010,100)
N = [1000]
K = 2       # Cantidad de bloques
O = 0.5       # Solapamiento

realizaciones = 200

# Parametros del ruido normal
u = 0   # Media
v = 2   # Varianza

resultados = []
tus_resultados = []

for Ni in N:
    
    #Determino el largo de cada bloque
    #Ln = np.floor(Ni/K)
    #D = np.floor(Ln*O)
    #L = Ln+D     
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
                
        # Aplicando una ventana al bloque
        w = np.reshape(bartlett(L),(L,1))        #Hacer el reshape dentro de la funcion de la ventana
        xw = x[Ki*D:Ki*D+L,:]*w
        
        print("Bloque desde:"+str(Ki*D)+" Hasta:"+str(Ki*D+L))
        #plt.figure(Ki)
        #plt.plot(np.arange(L),xw)
        #plt.plot(np.arange(L),w)
    
        espectroW = fft(xw,axis=0)*(1/L)
        
        PSDW = np.abs(espectroW)**2
        promPSDW = promPSDW+PSDW/K
    
     
    # Varianza del estimador
    varPSDW = np.var(promPSDW,axis=1)*(L**2)
    meanVarPSDW = np.mean(varPSDW)
    
    # Valor esperado de la PSD
    EPSDW = np.mean(promPSDW,axis=1)
    
    # Calculo el sesgo
    #sesgoW = v-np.sum(EPSDW)
    sesgoW = v/Ni-np.mean(EPSDW)
    
    resultados.append([sesgoW,meanVarPSDW])
    print("N: "+str(Ni)+", Sesgo: "+str(sesgoW)+", Varianza:"+str(meanVarPSDW))


#tus_resultados = [resultados[0],resultados[5-2],resultados[10-1],resultados[25-1],resultados[50-1],resultados[100-1],resultados[500-1]]


'''
grafResultados = np.array(resultados)
N = np.arange(100,5010,100)

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.title("Módulo del sesgo")
plt.plot(N,np.abs(grafResultados[:,0]),label="sin ventana")    
#plt.plot(N,np.abs(grafResultados[:,2]),label="FlatTop")       
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)

plt.subplot(212)
plt.title("Varianza")
plt.plot(N,grafResultados[:,1],label="sin ventana")
#plt.plot(N,grafResultados[:,3],label="FlatTop")
plt.axis([-10,N[-1]+10,0,1.5*v**2])
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
'''
