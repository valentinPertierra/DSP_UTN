# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 02:46:01 2018

@author: Valentin
"""

import numpy as np
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig

import scipy.io as sio
from scipy.fftpack import fft

from funciones import *
from time import time
from scipy.interpolate import CubicSpline


mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = np.reshape(ecg_one_lead,len(ecg_one_lead))

posQRS = mat_struct['qrs_detections']
posQRS = np.reshape(posQRS,len(posQRS))

N = len(ecg_one_lead)

fs = 1000
fn = fs/2

qrs = np.ones(len(posQRS))

# Grafico el ecg con las posiciones de las ondar qrs
plt.figure(1)
plt.plot(np.arange(N),ecg_one_lead)
plt.plot(posQRS,qrs,'o')
plt.grid()

# Asumo que el nivel isoeléctrico esta 70ms antes del latido 
# Genero una matriz con los niveles isoeléctricos, no tomo un solo punto, si no
# que voy a promediar determinada cantidad L de puntos que se encuentren en el intervalo PQ
L = 30
dt = 70

nIso = []
for i in posQRS:
  nIso.append(ecg_one_lead[i-dt-(L//2):i-dt+(L//2)])
  
nIso = np.array(nIso)

# Promedio los segmentos
nIso = np.transpose(np.mean(nIso, axis=1))

# Interpolo los puntos para obtener el estimador de la señal interferente
cs = CubicSpline(posQRS-dt, nIso)
estBni = cs(np.arange(N))

# Obtengo el estimador x del ecg
estXni = ecg_one_lead-estBni

plt.figure(2)
plt.plot(np.arange(N),ecg_one_lead,label = "Señal de ECG original")
plt.plot(np.arange(N),estBni,label = "Estimador B de la interferente")
plt.plot(np.arange(N),estXni,label = "Estimador x del ECG")
plt.plot(posQRS,qrs,'o',label = "Posicion de los latidos")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()



















