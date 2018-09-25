# -*- coding: utf-8 -*-
"""
Genero todas las señales del ejercicio 4
"""
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from funciones import *

#%%
def signal_4_a(fs,N):
    
    # Parametros de la senoidal
    a0 = np.sqrt(2)     #senoidal con energia unitaria
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la senoidal
    return generador_senoidal(fs,f0,N,a0,p0)

#%%
def signal_4_b(fs,N,a0=4.242727):
    
    # Parametros de la senoidal
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la senoidal
    tt,signal = generador_senoidal(fs,f0,N,a0,p0)
    
    # Me quedo con un periodo de la senoidal
    signal[int(fs/f0):]=0
    
    return tt,signal

#%%
def signal_4_c(fs,N,a0=4.242727):
    
    # Parametros de la senoidal
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la senoidal
    tt,signal = generador_senoidal(fs,f0,N,a0,p0)
    
    # Pongo en cero los dos primeros ciclos, dejo uno y pongo el resto en cero
    signal[:int(2*fs/f0)+1]=0       #le sume 1 por al redondear la señal no quedaba bien
    signal[int(3*fs/f0):]=0
    
    return tt,signal


#%%
def signal_4_d(fs,N,a0=2.9105050):
    
    # Parametros de la 1° senoidal
    a1 = a0     #senoidal con energia unitaria
    p1 = 0
    f1 = 9*fs/N
    
    # Parametros de la 2° senoidal
    a2 = a0     #senoidal con energia unitaria
    p2 = 0
    f2 = 8*fs/N
    
    # Genero la senoidal 1 
    tt,signal_1 = generador_senoidal(fs,f1,N,a1,p1)
    
    # Me quedo con un periodo de la senoidal
    signal_1[int(fs/f1):]=0
    
    # Genero la senoidal 2 
    tt,signal_2 = generador_senoidal(fs,f2,N,a2,p2)
    
    # Pongo en cero los dos primeros ciclos, dejo uno y pongo el resto en cero
    signal_2[:int(2*fs/f2)+1]=0       #le sume 1 por al redondear la señal no quedaba bien
    signal_2[int(3*fs/f2):]=0
    
    #sumo las senoidales
    signal = signal_1 + signal_2
    
    return tt,signal

#%%
def signal_4_e(fs,N,a0=2.9105050):
    
    # Parametros de la 1° senoidal
    a1 = a0     #senoidal con energia unitaria
    p1 = 0
    f1 = 8*fs/N
    
    # Parametros de la 2° senoidal
    a2 = a0     #senoidal con energia unitaria
    p2 = 0
    f2 = 9*fs/N
    
    # Genero la senoidal 1 
    tt,signal_1 = generador_senoidal(fs,f1,N,a1,p1)
    
    # Me quedo con un periodo de la senoidal
    signal_1[int(fs/f1):]=0
    
    # Genero la senoidal 2 
    tt,signal_2 = generador_senoidal(fs,f2,N,a2,p2)
    
    # Pongo en cero los dos primeros ciclos, dejo uno y pongo el resto en cero
    signal_2[:int(2*fs/f2)+1]=0       #le sume 1 por al redondear la señal no quedaba bien
    signal_2[int(3*fs/f2):]=0
    
    #sumo las senoidales
    signal = signal_1 + signal_2
    
    return tt,signal

#%%
def signal_4_f(fs,N,a0=2.4494949):
    
    # Parametros de la senoidal
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la senoidal  
    tt,signal = generador_senoidal(fs,f0,N,a0,p0)
    
    # Me quedo con 3 periodos de la senoidal
    signal[int(3*fs/f0):]=0
    
    return tt,signal

#%%
def signal_4_g(fs,N):
    
    # Parametros de la senoidal
    a1 = 0.5
    a2 = 4.09282828     #Para que este normalizada la energia
    a3 = 1
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la 1 senoidal  
    tt,signal_1 = generador_senoidal(fs,f0,N,a1,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_1[int(1*fs/f0):]=0
    
    
    # Genero la 2 senoidal  
    tt,signal_2 = generador_senoidal(fs,f0,N,a2,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_2[int(2*fs/f0):]=0
    signal_2[:int(1*fs/f0)+1]=0
    
    # Genero la 3 senoidal  
    tt,signal_3 = generador_senoidal(fs,f0,N,a3,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_3[int(3*fs/f0):]=0
    signal_3[:int(2*fs/f0)]=0
    
    #sumo las senoidales
    signal = signal_1 + signal_2 + signal_3
    
    return tt,signal

#%%
def signal_4_h(fs,N):
    
    # Parametros de la senoidal
    a1 = 0.25
    a2 = 2.31842424     #Para que este normalizada la energia
    a3 = 0.75
    p0 = 0
    f0 = 9*fs/N
    
    # Genero la 1 senoidal  
    tt,signal_1 = generador_senoidal(fs,f0,N,a1,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_1[int(1*fs/f0):]=0
    
    
    # Genero la 2 senoidal  
    tt,signal_2 = generador_senoidal(fs,f0,N,a2,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_2[int(2*fs/f0):]=0
    signal_2[:int(1*fs/f0)+1]=0
    
    # Genero la 3 senoidal  
    tt,signal_3 = generador_senoidal(fs,f0,N,a3,p0)
    
    # Me quedo con 1 periodo de la senoidal
    signal_3[int(3*fs/f0):]=0
    signal_3[:int(2*fs/f0)]=0
    
    #sumo las senoidales
    signal_4 = signal_1 + signal_2 + signal_3
    
    # Me quedo con un periodo de signal (los tres periodos de las senoidales)
    # y lo repito tres veces
    signal = []
    signal[:] = signal_4[:int(3*fs/f0)]
    signal[int(3*fs/f0):] = signal_4[:int(3*fs/f0)]
    signal[int(6*fs/f0):] = signal_4[:int(3*fs/f0)+1]
    
    return tt,signal

#%%
def signal_4_i(fs,N,a0=3.00002020):
    
    # Parametros de la 1° senoidal
    a1 = a0     #senoidal con energia unitaria
    p1 = 0
    f1 = 9*fs/N
    
    # Parametros de la 2° senoidal
    a2 = a0     #senoidal con energia unitaria
    p2 = np.pi
    f2 = 9*fs/N
    
    # Genero la senoidal 1 
    tt,signal_1 = generador_senoidal(fs,f1,N,a1,p1)
    
    # Me quedo con un periodo de la senoidal
    signal_1[int(fs/f1):]=0
    
    # Genero la senoidal 2 
    tt,signal_2 = generador_senoidal(fs,f2,N,a2,p2)
    
    # Me quedo con un periodo de cada señal
    signal_2[:int(1*fs/f2)]=0       
    signal_2[int(2*fs/f2):]=0
    
    #sumo las senoidales
    signal = signal_1 + signal_2
    
    return tt,signal

#%%
# Señal cuadrada 
def signal_cuadrada(fs,N):
    
    a0 = 0.5
    D = 1/9
    T0 = (1/fs)*(N+1)
    
    tt,signal = generador_cuadrada(fs, 1/T0, N, a0, D)
    
    return tt,signal+a0
    


    







