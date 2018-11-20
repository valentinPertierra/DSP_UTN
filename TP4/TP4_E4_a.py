# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:27:51 2018

@author: Valentin
"""

import numpy as np
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig

import scipy.io as sio
from scipy.fftpack import fft
from scipy import signal

from funciones import *


def vertical_flaten(a):    
    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

fs = 1000
fn = fs/2

# Genero un vector con el tiempo
t = np.linspace(0,(N-1)/fs,N)

# Para determinar la plantilla del filtro, grafico la psd de las zonas del ECG 
# que contienen latidos sin señales interferentes y zonas afectadas por la interferente.
# De este forma determino que parte del espectro contiene señales interferentes

# Determino las regiones "buenas" y afectadas por interferentes
regionOk_1 = np.arange(30000,50000, dtype='uint')
regionOk_2 = np.arange(60000,80000, dtype='uint')
regionInt_1 = np.arange(100000,120000, dtype='uint')
regionInt_2 = np.arange(725000,745000, dtype='uint')
regionInt_3 = np.arange(920000,940000, dtype='uint')

# Obtengo las psd con el metodo de Welch
L = 10000 #Largo de los bloques a promediar

psdWelch = np.zeros(shape=((L//2)+1,5))
f,psdWelch[:,0] = sig.welch(ecg_one_lead[regionOk_1,0], fs, 'bartlett',nperseg=L, noverlap=L//2)
f,psdWelch[:,1] = sig.welch(ecg_one_lead[regionOk_2,0], fs, 'bartlett',nperseg=L, noverlap=L//2)
f,psdWelch[:,2] = sig.welch(ecg_one_lead[regionInt_1,0], fs, 'bartlett',nperseg=L, noverlap=L//2)
f,psdWelch[:,3] = sig.welch(ecg_one_lead[regionInt_2,0], fs, 'bartlett',nperseg=L, noverlap=L//2)
f,psdWelch[:,4] = sig.welch(ecg_one_lead[regionInt_3,0], fs, 'bartlett',nperseg=L, noverlap=L//2)

# Normalizo los resultados
psdWelch = psdWelch/np.max(psdWelch)


# Grafico las regiones del ECG 
plt.figure(1)
plt.plot(ecg_one_lead[regionOk_1,0],label = 'OK 1')
plt.plot(ecg_one_lead[regionOk_2,0],label = 'OK 2')
plt.plot(ecg_one_lead[regionInt_1,0],label = 'Con Interferente 1')
plt.plot(ecg_one_lead[regionInt_2,0],label = 'Con Interferente 2')
plt.plot(ecg_one_lead[regionInt_3,0],label = 'Con Interferente 3')
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()


# Grafico las psd regiones del ECG 
plt.figure(2)
plt.plot(f,20*np.log10(psdWelch[:,0]),label = 'OK 1')
plt.plot(f,20*np.log10(psdWelch[:,1]),label = 'OK 2')
plt.plot(f,20*np.log10(psdWelch[:,2]),label = 'Con Interferente 1')
plt.plot(f,20*np.log10(psdWelch[:,3]),label = 'Con Interferente 2')
plt.plot(f,20*np.log10(psdWelch[:,4]),label = 'Con Interferente 3')
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.axis([0,25,-150,0])
plt.grid()


# Calculo donde esta el 90% de la energia de la señal OK
enTot = np.sum(psdWelch[:,0])
en90 = 0.9*enTot

suma = 0
for i in range(len(psdWelch[:,0])):
    suma += psdWelch[i,0]
    if suma >= en90:
        frec90 = f[i]
        break



'''
secESG = ecg_one_lead[]
esp_ECG_ok = fft(secESG)*(1/N)
esp_ECG_ok = esp_ECG_ok[:int(N/2)]

# Grafico la respuesta de la señal
plt.figure(1)
plt.plot(fn*(w/np.pi),20*np.log10(np.abs(h)))
plt.grid()
'''


