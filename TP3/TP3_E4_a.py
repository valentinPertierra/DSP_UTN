# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:19:15 2018

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
from scipy import signal

# Periodograma de Welch casero
def periodogramaWelch(x,K,overlap):
    
    # Determino el largo de la señal y la cantidad de realizaciones
    N = len(x[:,0])
    R = len(x[0,:])
    
    # Calculo el largo de cada bloque
    L = N/(1+(1-O)*(K-1))
    
    D = (1-O)*L 
    D = np.round(D)
    L = N-D*(K-1)
    
    D = int(D)
    L = int(L)
    
    # Inicializo vectores
    promPSDW = np.zeros(shape=(L//2,R))
    xw = np.zeros(shape=(L,R))
    
    # Promedio los PSD de los K bloques solapados
    for Ki in range(K):

        w = np.reshape(bartlett(L),(L,1))        
        xw = x[Ki*D:Ki*D+L,:]*w
               
        espW = fft(xw,axis=0)*(1/L)
        
        PSDW = np.abs(espW)**2
        PSDW = PSDW[:L//2]
        promPSDW = promPSDW+PSDW/K
        
        
    # Genero vector con frecuencias
    f = np.arange(L//2)*(fs/L)
    
    return f,promPSDW


# Obtené los valores XX para que cumplas con el enunciado
SNR = np.array([-3,-10])

fs = 1000 # Hz
N = 1000
df = fs/N

# Parametros para el metodo de Welch
K = 2       # Cantidad de bloques
O = 0.5     # Solapamiento

# Realizaciones
R = 200

# Parametros de la señal
a1 = np.sqrt(2)   # Esta normalizada en energia
f0 = fs/4

# Variable aleatoria con distribución uniforme para fr
va = np.random.uniform(-1/2,1/2,R)

# Genero un vector con el tiempo
t = np.linspace(0,(N-1)/fs,N)

# Genero la señal senoidal
x = []
for fr in va:
    f = f0+fr*df   
    x.append(a1*np.sin(t*2*np.pi*f))

x1 = np.transpose(np.array(x))

psd = []
resultados = []
for db in SNR:
    
    # Parametros del ruido normal
    u = 0                       # Media
    v = (N/2)*10**(db/10)        # Varianza
    
    # Genero señal de ruido
    n = np.sqrt(v)*np.random.randn(N,R)+u

    # Le sumo el ruido a la señal senoidal
    x = x1+n

    # Obtengo el periodograma de Welch
    f,psdWelch = periodogramaWelch(x,K,O)
    
    # Estimo f0 para cada realización
    estF0 = f[np.argmax(psdWelch,axis=0)]
    
    # Obtengo el valor esperado de f0 y la varianza
    valEspF0 = np.mean(estF0)
    varEstF0 = np.var(estF0)
    
    psd.append(psdWelch)
    resultados.append([valEspF0,varEstF0])

#plt.figure(figsize=(10,15))
plt.figure()
plt.subplot(221)
plt.title("PSD de las distintas realizaciones con SNR=3db")
plt.plot(f,20*np.log10(psd[0]))          
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(222)
plt.title("Promedio de las distintas realizaciones con SNR=3db")
plt.plot(f,20*np.log10(np.mean(psd[0],axis = 1)))          
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(223)
plt.title("PSD de las distintas realizaciones con SNR=10db")
plt.plot(f,20*np.log10(psd[1]))          
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(224)
plt.title("Promedio de las distintas realizaciones con SNR=10db")
plt.plot(f,20*np.log10(np.mean(psd[1],axis = 1)))          
plt.xlabel('f [Hz]')
plt.grid(True)

print(resultados)