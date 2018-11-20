# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:29:59 2018

@author: Valentin
"""

## Inicialización del Notebook del TP4

import numpy as np
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig

import scipy.io as sio
from scipy.fftpack import fft

from funciones import *


def vertical_flaten(a):    
    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('./ECG_TP4.mat')

#ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = np.reshape(ecg_one_lead,len(ecg_one_lead))

N = len(ecg_one_lead)

fs = 1000
fn = fs/2


#ecg_one_lead_med200 = sig.medfilt(ecg_one_lead[80000:12000],201)
#ecg_one_lead_med600 = sig.medfilt(ecg_one_lead_med200,601)
#Nm = len(ecg_one_lead_med200)

'''
ecg_one_lead_med200 = sig.medfilt(ecg_one_lead,199)
ecg_one_lead_med600 = sig.medfilt(ecg_one_lead_med200,599)

ecg_one_lead_dif = ecg_one_lead - ecg_one_lead_med600
'''
print(np.mean(ecg_one_lead_med600))
estB = ecg_one_lead_med600 - np.mean(ecg_one_lead_med600)
esp_B = fft(estB)/N


plt.figure(1)
plt.plot(np.arange(N),ecg_one_lead,label = 'original')
plt.plot(np.arange(N),ecg_one_lead_med200,label = 'filtrada 200')
plt.plot(np.arange(N),ecg_one_lead_med600,label = 'filtrada 600')
plt.plot(np.arange(N),ecg_one_lead_dif,label = 'diferencia')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.show()



plt.figure(2)
plt.plot(np.arange(N),ecg_one_lead_med600,label = 'filtrada 600')
plt.plot(np.arange(N),ecg_one_lead_dif,label = 'diferencia')
plt.show()

plt.figure(3)
plt.plot(np.arange(N//2)*(fs/N),np.abs(esp_B[0:N//2])**2,label = 'espectro')
plt.show()

plt.figure(4)
plt.plot(np.arange(N),ecg_one_lead_med600,label = 'filtrada 600')
plt.plot(np.arange(N),estB,label = 'sin valor medio')
plt.show()

