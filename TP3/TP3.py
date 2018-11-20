# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:27:54 2018

@author: Valentin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML

from funciones import *

N = 1024
fs = 1000

realizaciones = 200

# Parametros del ruido normal
u = 0
v = 3

# Vectores donde guardo las distintas realizaciones
realizacion = []

# Hago las realizaciones
for i in range(realizaciones):
    
    # Genero la señal de ruido normal
    t,x = genRuidoNormal(u,v,N,fs)
    
    # Obtengo el espectro
    f,eX = analizadorEspectro(x,t)
    
    # Guardo el modulo del espectro al cuadrado (DEP)
    realizacion.append(np.abs(eX)**2)
   
    
realizacion = np.array(realizacion)

varianza = np.var(realizacion,axis=0)*(N**2)
media = np.mean(realizacion,axis=0)

madiaVar = np.mean(varianza)

areaMed = np.sum(media)

print(areaMed)
#print(media)
#print(varianza)

print(madiaVar)



plt.figure(0)
plt.plot(f, realizacion[0,:])     
#plt.plot(t,x)       
plt.grid(True)
plt.show()




































'''
a0Dirichlet = np.abs(eDirichlet[:,N//4])
a0Bartlett = np.abs(eBartlett[:,N//4])
a0Hann = np.abs(eHann[:,N//4])
a0Blackman = np.abs(eBlackman[:,N//4])
a0FlatTop = np.abs(eFlatTop[:,N//4])
'''

'''
# Genero un vector con los valores del estimador de a0 para las distintas señales
a0Dirichlet = []
a0Bartlett = []
a0Hann = []
a0Blackman = []
a0FlatTop = []

for i in range(realizaciones):
    a0Dirichlet.append(np.abs(eDirichlet[i][N//4]))
    a0Bartlett.append(np.abs(eBartlett[i][N//4]))
    a0Hann.append(np.abs(eHann[i][N//4]))
    a0Blackman.append(np.abs(eBlackman[i][N//4]))
    a0FlatTop.append(np.abs(eFlatTop[i][N//4]))
  '''  
    
    
'''    
plt.figure(figsize=(16,22))
plt.subplot(511)
plt.title("Histograma de |$X^0_w(\Omega_0)$| utilizando el kernel de Dirichlet")
plt.hist(a0Dirichlet, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(512)
plt.title("Histograma de |$X^1_w(\Omega_0)$| utilizando la ventana Bartlett")
plt.hist(a0Bartlett, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(513)
plt.title("Histograma de |$X^2_w(\Omega_0)$| utilizando la ventana Hann")
plt.hist(a0Hann, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(514)
plt.title("Histograma de |$X^3_w(\Omega_0)$| utilizando la ventana Blackman")
plt.hist(a0Blackman, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(515)
plt.title("Histograma de |$X^4_w(\Omega_0)$| utilizando la ventana FlatTop")
plt.hist(a0FlatTop, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.show()

'''



'''
# Ejercicio 3_b

a = N//4-2
b = N//4+3


a0Dirichlet = np.abs(eDirichlet[:,a:b]/2)**2
a0Bartlett = np.abs(eBartlett[:,a:b]/2)**2
a0Hann = np.abs(eHann[:,a:b]/2)**2
a0Blackman = np.abs(eBlackman[:,a:b]/2)**2
a0FlatTop = np.abs(eFlatTop[:,a:b]/2)**2

a0Dirichlet = np.sqrt(np.sum(a0Dirichlet, axis=1)*(1/5))
a0Bartlett = np.sqrt(np.sum(a0Bartlett, axis=1)*(1/5))
a0Hann = np.sqrt(np.sum(a0Hann, axis=1)*(1/5))
a0Blackman = np.sqrt(np.sum(a0Blackman, axis=1)*(1/5))
a0FlatTop = np.sqrt(np.sum(a0FlatTop, axis=1)*(1/5))

plt.figure(figsize=(16,22))
plt.subplot(511)
plt.title("Histograma de |$X^0_w(\Omega_0)$| utilizando el kernel de Dirichlet")
plt.hist(a0Dirichlet, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(512)
plt.title("Histograma de |$X^1_w(\Omega_0)$| utilizando la ventana Bartlett")
plt.hist(a0Bartlett, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(513)
plt.title("Histograma de |$X^2_w(\Omega_0)$| utilizando la ventana Hann")
plt.hist(a0Hann, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(514)
plt.title("Histograma de |$X^3_w(\Omega_0)$| utilizando la ventana Blackman")
plt.hist(a0Blackman, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(515)
plt.title("Histograma de |$X^4_w(\Omega_0)$| utilizando la ventana FlatTop")
plt.hist(a0FlatTop, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.show()


'''