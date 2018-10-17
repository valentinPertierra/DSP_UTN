# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:44:11 2018

@author: Valentin
"""
'''
TP2: Ventanas 
Ejercicio 3

'''
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from funciones import *
from scipy import signal

N = 1024
fs = 1000

realizaciones = 200

# Parametros de la señal
a0 = 2
p0 = 0
f0 = fs/4


#f01 = fs/4+d1*(fs/N)

# Variable aleatoria con diatribución uniforme
variableAleatoria  = np.random.uniform(-2,2,realizaciones)

# Estimador de a0
estA0 = []

# Hago tantas realizaciones como variables aleatorias tenga
for fr in variableAleatoria:
    
    # Determino f1
    f1 = f0+fr*(fs/N)
    
    # Genero la señal senoidal
    tt,x = generador_senoidal(fs,f1,N,a0,p0)
    
    # Genero ventaneando la señal
    #ventana = signal.blackmanharris(N)
    #ventana = signal.flattop(N)
    #x = x*ventana

    # Obtengo el espectro
    ff,X = analizadorEspectro(x)
    
    estA0.append(np.abs(X[N//4]))
    
# Grafico el histograma del estimador
histograma(estA0,10)
mediaA0 = getValorMedio(estA0)
sesgoA0 = mediaA0-a0
varianzaA0 = getDesvioEstandar(estA0)**2

print("Media = %f"%mediaA0)
print("Varianza = %f"%varianzaA0)
print("Sesgo = %f"%sesgoA0)

# Estimador de a01
estA1 = []
a = N//4 - 2
b = N//4 + 2

aux = []

# Hago tantas realizaciones como variables aleatorias tenga
for fr in variableAleatoria:
    
    # Determino f1
    f1 = f0+fr*(fs/N)
    
    # Genero la señal senoidal
    tt,x = generador_senoidal(fs,f1,N,a0,p0)

    # Obtengo el espectro
    ff,X = analizadorEspectro(x,tt)
    aux = X[a:b+1]
    
    valorRMS = 0
    for x in aux:
        valorRMS += np.abs(x/2)**2
    
    valorRMS = np.sqrt(valorRMS*(1/5))
    estA1.append(valorRMS)
    

# Grafico el histograma del estimador
histograma(estA1,10)
mediaA1 = getValorMedio(estA1)
varianzaA1 = getDesvioEstandar(estA1)**2
sesgoA1 = mediaA1-(a0/2)

print("Media = %f"%mediaA1)
print("Varianza = %f"%varianzaA1)
print("Sesgo = %f"%sesgoA1)















