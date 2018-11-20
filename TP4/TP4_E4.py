# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:55:17 2018

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

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

fs = 1000
fn = fs/2

# Defina la plantilla del filtro
fs0 = 0.25 # fin de la banda de detenida 0
fc0 = 0.5 # comienzo de la banda de paso
fc1 = 15 # fin de la banda de paso
fs1 = 17 # comienzo de la banda de detenida 1

# Atenuacion en la banda de paso y de stop en dB
attPaso = 0.5
attStop = 20.0

fecuencias = np.array([0.0,fs0+1,fc0+1,fc1,fs1,fn])/fn
ganancias = np.array([-attStop*10, -attStop*10, -attPaso, -attPaso, -attStop*10, -attStop*10])
ganancias = 10**(ganancias/20)

# Obtengo los parametros de los distintos filtros
bpSosButter = sig.iirdesign(wp=np.array([fc0,fc1])/fn,ws=np.array([fs0,fs1])/fn,gpass=attPaso,gstop=attStop,analog=False,ftype='butter',output='sos')
bpSosCheby1 = sig.iirdesign(wp=np.array([fc0,fc1])/fn,ws=np.array([fs0,fs1])/fn,gpass=attPaso,gstop=attStop,analog=False,ftype='cheby1',output='sos')

numFIR1 = sig.firwin2(1001, fecuencias,ganancias, window = 'bartlett')
numFIR2 = sig.firwin2(501, fecuencias,ganancias, window = 'blackmanharris')
denFIR = 1.0

wButter,hButter = sig.sosfreqz(bpSosButter, worN=1500)
wCheby1,hCheby1 = sig.sosfreqz(bpSosCheby1, worN=1500)
wFIR1,hFIR1 = sig.freqz(numFIR1, denFIR)
wFIR2,hFIR2 = sig.freqz(numFIR1, denFIR)

# Grafico la respuesta del filtro
plt.figure(1)
plt.plot(fn*(wButter/np.pi),20*np.log10(np.abs(hButter)),label = 'Butter')
plt.plot(fn*(wCheby1/np.pi),20*np.log10(np.abs(hCheby1)),label = 'Cheby1')
plt.plot(fn*(wFIR1/np.pi),20*np.log10(np.abs(hFIR1)),label = 'FIR1')
plt.plot(fn*(wFIR2/np.pi),20*np.log10(np.abs(hFIR2)),label = 'FIR2')
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()

ecg_one_lead = np.reshape(ecg_one_lead,N)

# Filtro la señal de ECG
ECG_butter = sig.sosfiltfilt(bpSosButter, ecg_one_lead) 
ECG_Cheby1 = sig.sosfiltfilt(bpSosCheby1, ecg_one_lead) 
ECG_FIR1 = sig.filtfilt(numFIR1, denFIR, ecg_one_lead)
ECG_FIR2 = sig.filtfilt(numFIR2, denFIR, ecg_one_lead)


region = np.arange(12*60*fs, 12.4*60*fs,dtype='uint') 

# Grafico la señal original y la filtrada
plt.figure(2)
plt.plot(np.arange(N),ecg_one_lead,label = 'Original')
plt.plot(np.arange(N),ECG_butter,label = 'Butter')
plt.plot(np.arange(N),ECG_Cheby1,label = 'Cheby1')
plt.plot(np.arange(N),ECG_FIR1,label = 'FIR1')
plt.plot(np.arange(N),ECG_FIR2,label = 'FIR2')
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()







