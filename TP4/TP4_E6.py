# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:56:03 2018

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

# Patron de latidos normales
hb_1 = mat_struct['heartbeat_pattern1']
hb_1 = np.reshape(hb_1,len(hb_1))

# Patron de latidos ventriculares
hb_2 = mat_struct['heartbeat_pattern2']
hb_2 = np.reshape(hb_2,len(hb_2))

# Patron del complejo QRS
qrs_1 = mat_struct['qrs_pattern1']
qrs_1 = np.reshape(qrs_1,len(qrs_1))

# Ubicacion conocda de los latidos 
posQRS = mat_struct['qrs_detections']
posQRS = np.reshape(posQRS,len(posQRS))

# Prefiltro la señal de ECG con algun metodo:


N = len(ecg_one_lead)

fs = 1000
fn = fs/2

qrs = np.ones(len(posQRS))


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


#     ACA ARRANCA EL EJERCICIO 6
# Grafico los patrones
'''
plt.figure(1)
plt.plot(hb_1,label = "Patron latido normal")
plt.plot(hb_2,label = "Patron latido ventricular")
plt.plot(qrs_1,label = "Patron qrs")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()
'''

'''
# Hago la correlacion cruzada entre el patron y la señal original filtrada
corr_hb_1 = sig.correlate(estXni, hb_1, mode='same')/N
corr_hb_2 = sig.correlate(estXni, hb_2, mode='same')/N
corr_qrs_1 = sig.correlate(estXni, qrs_1, mode='same')/N


plt.figure(2)
plt.plot(np.arange(N),estXni,label = "Señal de ECG sin interferente")
plt.plot(np.arange(N),corr_hb_1,label = "Correlacion con patron latido normal")
plt.plot(np.arange(N),corr_hb_2,label = "Correlacion con patron latido ventricular")
plt.plot(np.arange(N),corr_qrs_1,label = "Correlacion con patron qrs")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()

'''
# Doy vuelta las señales patron 
#hb_1_flip = np.flipud(hb_1)
hb_2_flip = np.flipud(hb_2)
#hb_qrs_flip = np.flipud(qrs_1)

# Hago la convolucion
#conv_hb_1 = np.convolve(estXni,hb_1_flip,'same')/(N//4)
conv_hb_2 = np.convolve(estXni,hb_2_flip,'same')/N
#conv_qrs = np.convolve(estXni,hb_qrs_flip,'same')/N


# Desplazo los resultados para que los maximos coincidan con los latidos
#conv_hb_1 = np.roll(conv_hb_1,-164)
conv_hb_2 = np.roll(conv_hb_2,-91)
#conv_qrs = np.roll(conv_qrs,-len(hb_qrs_flip)//2)


qrs = np.ones(len(posQRS))/4

'''
plt.figure(3)

plt.plot(np.arange(N),estXni,label = "Señal de ECG sin interferente")
#plt.plot(np.arange(N),conv_hb_1,label = "Convolucion con patron latido normal")
plt.plot(np.arange(N),conv_hb_2,label = "Convolucion con patron latido ventricular")
#plt.plot(np.arange(N),conv_qrs,label = "Convolucion con patron qrs")
plt.plot(posQRS,qrs,'o',label = "ubucacion latidos")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()
'''

# Algoritmo para la deteccion de latidos ventriculares
# Normlizo la secucuencia obtenida luego del filtro adaptado y lo elevo al cuadrado
conv_hb_2 = conv_hb_2/np.max(conv_hb_2)
conv_hb_2 = conv_hb_2**2
      
TIME_OFF = 300   # Tiempo que no detecta latidos en ms despus de detectar uno
cont = 0
posVentr = []
maxLocal = 0
trh = 0.1
flag = 0
for i in range(len(conv_hb_2)):
    
    if flag != 1:
        if conv_hb_2[i]>trh:
            posVentr.append(i)
            flag = 1
    else:
        cont = cont + 1
        if cont == TIME_OFF:
            cont = 0
            flag = 0
            
        
posVentr = np.transpose(np.array(posVentr))

ven = np.zeros(len(posVentr))

# Grafico todos los latidos ventriculares detectados
'''
plt.figure(4)
plt.plot(np.arange(N),conv_hb_2,label = "Convolucion con patron latido ventricular")
plt.plot(posVentr,ven,'o',label = "ubucacion latidos ventricular")
plt.plot(posQRS,qrs,'o',label = "ubucacion todos lso latidos")
plt.grid()
'''

# Elimino los latidos ventriculares detectador para que no me interfieran con
# la deteccion de los latidos normales
ecgSinVentr = estXni
dt = 700     #ancho del pulso ventricular

for i in posVentr:
    ecgSinVentr[i-dt//2:i+dt//2] = 0

# Grafico el ecg sin los latidos ventriculares
'''
plt.figure(5)
plt.plot(np.arange(N),ecgSinVentr,label = "Convolucion con patron latido ventricular")
plt.plot(posVentr,ven,'o',label = "ubucacion latidos ventricular")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()
'''
# Detecto los latidos normales
# Doy vuelta las señales patron 
#hb_1_flip = np.flipud(hb_1)
hb_qrs_flip = np.flipud(qrs_1)

# Hago la convolucion
#conv_hb_1 = np.convolve(ecgSinVentr,hb_1_flip,'same')/(N//4)
#conv_qrs = np.convolve(ecgSinVentr,hb_qrs_flip,'same')/N
conv_qrs = np.convolve(estXni,hb_qrs_flip,'same')/N

# Desplazo los resultados para que los maximos coincidan con los latidos
#conv_hb_1 = np.roll(conv_hb_1,-164)
conv_qrs = np.roll(conv_qrs,-1)

# Normlizo la secucuencia y la elevo al cuadrado
#conv_hb_1 = conv_hb_1/np.max(conv_hb_1)
#conv_hb_1 = conv_hb_1**2

conv_qrs = conv_qrs/np.max(conv_qrs)
conv_qrs = conv_qrs**2

conv_hb_1 = conv_qrs

# Hago la deteccion de los latidos normales
TIME_OFF = 300   # Tiempo que no detecta latidos en ms despus de detectar uno
cont = 0
posNorm = []
trh = 0.12
flag = 0
for i in range(len(conv_hb_1)):
    
    if flag != 1:
        if conv_hb_1[i]>trh:
            posNorm.append(i)
            flag = 1
    else:
        cont = cont + 1
        if cont == TIME_OFF:
            cont = 0
            flag = 0

posNorm = np.transpose(np.array(posNorm))

norm = np.zeros(len(posNorm))
 

# Grafico el resultado
'''
plt.figure(6)
plt.plot(np.arange(N),conv_hb_1,label = "Convolucion con patron latido ventricular")
plt.plot(posVentr,ven,'o',label = "ubucacion latidos ventricular")
plt.plot(posNorm,norm,'o',label = "ubucacion latidos normales")
plt.plot(posQRS,qrs,'o',label = "ubucacion latidos")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()
'''

# Genero un array que contenga todos los latidos detectados, ventriculares y normales
detec = np.concatenate((posVentr,posNorm),axis=0)

# Acomodo el array 
detec.sort()

# Genero un array con unos en las posiciones detectadas
detect = np.zeros(N)
for i in detec:
    detect[i] = 1

'''
plt.figure(7)
plt.plot(posQRS,qrs,'o',label = "ubucacion latidos")
plt.plot(np.arange(N),detect,label = "latidos detectados")
plt.legend(bbox_to_anchor=(0.85, 0.98), loc=2, borderaxespad=0.)
plt.grid()
'''

# Verdadero Positivo (VP): yo digo que hay un latido y en realidad hay
# Falso Negativo (FN): yo digo que no hay un latido pero en realidad hay 
# Falso Positivo (FP): yo digo que hay un latido pero en realidad no hay 

# Cuento cunatos latidos son verdaderos positivos y falso negativos
VP = 0
FN = 0
dt = 35
for i in posQRS:
    if np.sum(detect[i-dt:i+dt]) == 1:
        VP = VP+1
    else:
        FN = FN+1
        

# Genero un array con unos en las posiciones donde efectivamente hay un latido
detectQRS = np.zeros(N)
for i in posQRS:
    detectQRS[i] = 1
 

FP = 0
for i in posNorm:
    if np.sum(detectQRS[i-dt:i+dt]) == 0:
        FP = FP+1

print("Error maximo de tiempo (dt): "+str(dt)+"ms")
print("Verdaderos positivos (VP): "+str(VP))
print("Falso Negativo (FN): "+str(FN))
print("Falso Positivo (FP): "+str(FP))



