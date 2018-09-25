# -*- coding: utf-8 -*-
"""
Generador de señales
"""
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# los % son directivas al IPython
# Borro todas las variables del workspace
#%reset 

#%%
""" 
brief:  Generador de señales senoidal, con argumentos
fs:     frecuencia de muestreo de la señal [Hz]
N:      cantidad de muestras de la señal a generar
f0:     frecuencia de la senoidal [Hz]
a0:     amplitud pico de la señal [V]
p0:     fase de la señal sinusoidal [rad]

como resultado la señal devuelve:

signal: senoidal evaluada en cada instante 
tt:     base de tiempo de la señal
"""
def generador_senoidal(fs, f0, N, a0=1, p0=0):
    
    # comienzo de la función
    # Genero un vector con el tiempo
    tt = np.linspace(0,(N-1)/fs,N)
    
    # Genero vector con los resultados
    signal = a0*np.sin(tt*2*np.pi*f0+p0)
        
    # fin de la función
    
    return tt, signal
    
    
#%%
'''  
Funcion para generar una señal de ruido blanco (gaussiano)  
N: cantidad de muestras
fs: frecuencia de muestreo
u: media
v: varianza
'''  
def genRuidoNormal(u,v,N,fs):
    
    # Genero un vector con el tiempo
    t = np.linspace(0,(N-1)/fs,N)
    
    # Genero un vector con la señal aleatoria de Nx1
    v = np.sqrt(v)
    y = v*np.random.randn(N,1)+u
    
    y = np.transpose(y)
    y = np.reshape(y,N)
    
    return t,y
    
#%%
""" 
Funcion para generar una señal cuadrada 
N: cantidad de muestras
fs: frecuencia de muestreo [Hz]
D: duty 0<D<1
f0: frecuencia [Hz]
a0: amplitud [V]
"""
def generador_cuadrada(fs, f0, N, a0, D):
    
    # comienzo de la función
    # Genero un vector con el tiempo
    tt = np.linspace(0,(N-1)/fs,N)
    
    #genero un periodo de la señal cuadrada
    T1 = a0* np.ones(int(D*fs/f0))
    T2 = -a0* np.ones(int((1-D)*fs/f0)) 
        
    T = np.concatenate((T1, T2), axis=None)
    
    # Caculo la cantidad de periodos segun la cantidad de muestras
    Np = np.ceil(N/len(T))
    
    signal = []
    for i in range(int(Np)):
        signal = np.concatenate((signal, T), axis=None)
    
    signal = signal[:N]
    
    # fin de la función
    
    return tt, signal
#%%
""" 
Funcion para generar una señal triangular
N: cantidad de muestras
fs: frecuencia de muestreo [Hz]
S: punto de simetria 0<D<1
f0: frecuencia [Hz]
a0: amplitud [V]
"""
def generador_triangular(fs, f0, N, a0, S):
    
    # comienzo de la función
    # Genero un vector con el tiempo
    tt = np.linspace(0,(N-1)/fs,N)
    
    #genero un periodo de la señal triangular
    T1 = S/f0
    tt1 = np.linspace(0,T1,fs*T1)
    y1 = (a0/T1)*tt1
    
    T2 = (1-S)/f0
    tt2 = np.linspace(0,T2,fs*T2)
    y2 = (-a0/T2)*tt2 + a0
    
    T = np.concatenate((y1, y2), axis=None)
    
    # Caculo la cantidad de periodos segun la cantidad de muestras
    Np = np.ceil(N/len(T))
    
    signal = []
    for i in range(int(Np)):
        signal = np.concatenate((signal, T), axis=None)
    
    signal = signal[:N]
    
    # fin de la función
    
    return tt, signal
#%%    
'''
Funcion para graficar las señales
'''
def graficar(t,y,figura,titulo,Xlabel,Ylabel):
    
    plt.figure(figura)
    plt.plot(t,y,label="señal")
    plt.title(titulo)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0,10,-2,2])  # [xi,xf,yi,yf]
    plt.grid(True)

#%%    
'''
Funcion para graficar el espectro
'''
def graficarEspectro(ff,espectro,figura,titulo):
    
    plt.figure(figura)
    plt.subplot(211)
    plt.title(titulo)
    plt.plot(ff, np.absolute(espectro))           #con stem grafico "barras"
    plt.xlabel('f [Hz]')
    plt.ylabel('|Y(f)|')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(ff, np.angle(espectro))
    plt.xlabel('f [Hz]')
    plt.ylabel('fase Y(f)')
    plt.grid(True)

#%%    
'''
Funcion para graficar el espectro
'''
def graficarEspectro_dB(ff,espectro,figura,titulo):
    
    plt.figure()
    plt.subplot(211)
    plt.title(titulo)
    plt.plot(ff, 20*np.log10(np.absolute(espectro)))           #con stem grafico "barras"
    plt.xlabel('f [Hz]')
    plt.ylabel('|Y(f)| [dB]')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(ff, np.angle(espectro))
    plt.xlabel('f [Hz]')
    plt.ylabel('fase Y(f)')
    plt.grid(True)

#%%
# Funcion para generar el espectro de una señal
def analizadorEspectro(yt,tt):

    #determino la frecuencia de muestreo
    Ts = tt[1]-tt[0]
    fs = 1/Ts

    #determino la cantidad de muestras de la señal (N)
    N = len(yt)

    # determino los valores de fecuencias
    df = fs/N
    f = np.arange(int(N/2))*df

    #aplico la fft a la señal y la normalizo
    yf = fft(yt)*(2/N)

    #me quedo con la mitad de las muestras
    yf = yf[:int(N/2)]

    return f,yf

#%% 
# Funcion para genero el histograma 
def histograma(yt,bin):
    
    plt.figure()
    plt.hist(yt, bin)
    plt.xlabel('N')
    plt.ylabel('histograma')
    #plt.axis([0,4,-2,2])
    plt.grid(True)
    plt.show()

    
#%%
# Cuantizador de señales
'''
La señal de entrada tiene que estear entre (-1,1) y devuelve las cuentas esta entre (-2^(bits-1),2^(bits-1)-1)
'''
def cuantizador(signal,bits,redondeo):
    
    # Dermino el maximo valor de cuentas
    cuentas = np.power(2,(bits-1))-1
    
    # Escalo la señal
    signalQ = signal*cuentas
    
    # Redondeo
    if redondeo == "ROUND":
        signalQ = np.round(signalQ)
    if redondeo == "FLOOR":
        signalQ = np.floor(signalQ)
    if redondeo == "CEIL":
        signalQ = np.ceil(signalQ)
    
    return signalQ

#%%
# Implementacion del algoritmo para calcular la DFT de una señal
def DFT(Xn):
    
    # Determino cuantas muestras tiene la señal
    N = len(Xn)
    
    Xk = np.zeros(N, dtype ='c16')
    
    for k in range(N-1):
        for n in range(N-1):
            Xk[k] += Xn[n]*np.exp(-1j*2*np.pi*k*n/N)
        
    return Xk

#%%
# Función para calcular la energía de una señal en el tiempo
def energiaTiempo(signal):
    
    energia=0
    for x in signal:
        energia += x**2
        
    energia = energia/len(signal)
    
    return energia

#%%
# Función para calcular la energía de una señal en el dominio frecuencial 
def energiaFrecuencia(espectro):
    
    energia = 0
    for x in espectro:
        energia += np.absolute(x/2)**2     
    
    energia = energia*2
    
    return energia    

#%%
# Función para calcular el valor medio de una señal
def getValorMedio(signal):
    
    valorMedio = 0
    for x in signal:
        valorMedio += x
    
    valorMedio = valorMedio/len(signal)
    
    return valorMedio

#%%
# Función para calcular el valor RMS de una señal a partir del espectro
def getValorRMS(espectro):
    
    valorRMS = 0
    for x in espectro:
        valorRMS += np.absolute(x/2)**2
    
    valorRMS = np.sqrt(valorRMS*2)
        
    return valorRMS

#%%
# Función para calcular la media de una señal
def getValorEsperado(signal):
    
    valorEsperado = 0
    for x in signal:
        valorEsperado += x
        
    valorEsperado = valorEsperado/len(signal)
        
    return valorEsperado

#%%
# Función para calcular la media de una señal
def getDesvioEstandar(signal):
    
    # Obtengo el valor esperado
    valorEsperado = 0
    for x in signal:
        valorEsperado += x
        
    valorEsperado = valorEsperado/len(signal)
    
    # Calculo el desvio estandar
    desvioEstandar = 0
    for x in signal:
        desvioEstandar += (x-valorEsperado)**2
        
    desvioEstandar = np.sqrt(desvioEstandar/len(signal))
        
    return desvioEstandar