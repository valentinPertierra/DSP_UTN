# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:32:44 2018

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

def vertical_flaten(a):    
    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = np.reshape(ecg_one_lead,len(ecg_one_lead))

N = len(ecg_one_lead)

fs = 1000
fn = fs/2


"""
# Con esto puedo calcular el tiempo que tarda en procesar
tStartMediana = time()

# Le aplico el filtro de mediana de 200 muestras
ecg_one_lead_med200 = sig.medfilt(ecg_one_lead,199)

# Le aplico el filtro de mediana de 600 muestras
# para obtener el estimador de la señal interferente
estB = sig.medfilt(ecg_one_lead_med200,599)

# Mido el timepo que tardo
dtMediana = time()-tStartMediana
print("Con el filtro de mediana tardo ="+str(dtMediana)+" segundos")

# Guardo estas variables en un archivo
np.savetxt('estB.dat', estB)
np.savetxt('dtMediana.dat', dtMediana)

# Para cargar los archivos se usa
# np.loadtxt('estB.dat')

"""
estB = np.loadtxt('estB.dat')
dtMediana = np.loadtxt('dtMediana.dat')

# Obtengo la señal de ECG sin la interferente
estX = ecg_one_lead - estB

# Determino donde se concentra la mayor cantidad de energía de la señal interferente
# Obtengo la psd utilizando el metodo de Welch
L = 10000 #Largo de los bloques a promediar

psdWelch = np.zeros(shape=((L//2)+1,5))
f,psdWelchEstB = sig.welch(estB, fs, 'bartlett',nperseg=L, noverlap=L//2)

# Normalizo la psd
psdWelchEstB = psdWelchEstB/np.max(psdWelchEstB)

# Grafico la psd
plt.figure(1)
plt.plot(f,20*np.log10(psdWelchEstB))
plt.grid()


# Determino la banda de frecuencia que contiene el 98% de la energía de la señal interferente
enTot = np.sum(psdWelchEstB)
en98 = 0.98*enTot

suma = 0
for i in range(len(psdWelchEstB)):
    suma += psdWelchEstB[i]
    if suma >= en98:
        frec98 = f[i]
        break
    
print("En f="+str(frec98)+"Hz se encuentra el 98% de la energía de la se señal")

# Adopto como nueva frecuencia de Nyquist fn2 = 10Hz  =>  fs2=20Hz
# Antes de diezmar la señal la tengo que filtrar para impedir que se meta alias
# Determino las caracteristicas del filtro digital
fp0 = 10
fs0 = 11
attPaso = 0.1
attStop = 100

# Obtengo los parametros del filtro
bpSosButter = sig.iirdesign(wp=fp0/fn,ws=fs0/fn,gpass=attPaso,gstop=attStop,analog=False,ftype='butter',output='sos')
wB,hB = sig.sosfreqz(bpSosButter, worN=1500)

# Grafico la respuesta del filtro
plt.figure(2)
plt.plot(fn*(wB/np.pi),20*np.log10(np.abs(hB)))
plt.grid()

tStartDiezmado = time()
# Filtro la señal original 
ecgB = sig.sosfiltfilt(bpSosButter, ecg_one_lead) 


plt.figure(3)
plt.plot(np.arange(N),ecg_one_lead,label= "Original")
plt.plot(np.arange(N),ecgB,label = "Filtrada")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()


# Diezmo la señal, la muestreo  a fs2 = 20Hz => 50 veces menos que fs1=1000Hz
fs2 = 20
fn2 = fs2/2

ecgD = ecgB[0:N:50]

N2 = len(ecgD)


# Le aplico el primer filtro de mediana 7 muestras
ecgD_med7 = sig.medfilt(ecgD,3)

# Le aplico el segundo filtro de mediana de 15 muestras
estBD = sig.medfilt(ecgD_med7,17)

# Grafico la señal
plt.figure(4)
plt.plot(np.arange(N2)*(fs2/N2),ecgD,label= "Original Diezmada")
plt.plot(np.arange(N2)*(fs2/N2),ecgD_med7,label = "1 filtro de mediana")
plt.plot(np.arange(N2)*(fs2/N2),estBD,label = "Interferente")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()

# Interpolo ceros en el estimador de la señal interferente obtenido
estBZ = np.zeros(N)
estBZ[0:N:50] = estBD

# Filtro la señal interpolada
estBmr = sig.sosfiltfilt(bpSosButter, estBZ) 

# Escalo la señal resultante
maxBZ = np.max(estBZ)
estBmr = maxBZ*estBmr/np.max(estBmr)

# Calculo el timepo de procesado
dtDiezmado = time()-tStartDiezmado
print("Con el diezmado tardo ="+str(dtDiezmado)+" segundos")

# Grafico el estimador de la señal interferente B, utlizando el filtrado no lineal, aplicando y no el diezmado 
plt.figure(5)
plt.plot(np.arange(N),estBmr,label= "Con MR")
plt.plot(np.arange(N),estBZ,label= "Interferente con interpolacion de ceros")
plt.plot(np.arange(N),estB,label= "Con filtro mediana sin MR")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()


# Le resto la interferente a la señal de ecg original
estXmr = ecg_one_lead - estBmr

# Grafico el resultado
plt.figure(6)
plt.plot(np.arange(N),ecg_one_lead,label= "ECG original")
plt.plot(np.arange(N),estXmr,label= "ECG sin interferente con tec. de multirate")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid()


