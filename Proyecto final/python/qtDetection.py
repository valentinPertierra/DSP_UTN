# -*- coding: utf-8 -*-
"""
@author: Valentin

"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig
import pywt

import scipy.io as sio
from scipy.fftpack import fft

#from funciones import *
from time import time


def vertical_flaten(a):    
    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = np.reshape(ecg_one_lead,len(ecg_one_lead))

# Ubicacion conocida de los latidos 
posQRS = mat_struct['qrs_detections']
posQRS = np.reshape(posQRS,len(posQRS))

N = len(ecg_one_lead)

fs = 1000
fn = fs/2

# Filtro la señal de interferente medinate un filtrado de mediana utilizando tecnicas de multirate
'''
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
#plt.figure(2)
#plt.plot(fn*(wB/np.pi),20*np.log10(np.abs(hB)))
#plt.grid()

# Filtro la señal original 
ecgB = sig.sosfiltfilt(bpSosButter, ecg_one_lead) 

#plt.figure(3)
#plt.plot(np.arange(N),ecg_one_lead,label= "Original")
#plt.plot(np.arange(N),ecgB,label = "Filtrada")
#plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
#plt.grid()

# Diezmo la señal, la muestreo  a fs2 = 20Hz => 50 veces menos que fs1=1000Hz
fs2 = 20
fn2 = fs2/2
ecgD = ecgB[0:N:50]
N2 = len(ecgD)

# Le aplico el primer filtro de mediana 5 muestras
ecgD_med7 = sig.medfilt(ecgD,3)

# Le aplico el segundo filtro de mediana de 15 muestras
estBD = sig.medfilt(ecgD_med7,17)

# Grafico la señal
#plt.figure(4)
#plt.plot(np.arange(N2)*(fs2/N2),ecgD,label= "Original Diezmada")
#plt.plot(np.arange(N2)*(fs2/N2),ecgD_med7,label = "1 filtro de mediana")
#plt.plot(np.arange(N2)*(fs2/N2),estBD,label = "Interferente")
#plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
#plt.grid()

# Interpolo ceros en el estimador de la señal interferente obtenido
estBZ = np.zeros(N)
estBZ[0:N:50] = estBD
estBZ[0] = 0

# Filtro la señal interpolada
estBmr = sig.sosfiltfilt(bpSosButter, estBZ) 

# Escalo la señal resultante
maxBZ = np.max(estBZ)
estBmr = maxBZ*estBmr/np.max(estBmr)

# Grafico el estimador de la señal interferente B, utlizando el filtrado no lineal, aplicando y no el diezmado 
#plt.figure(5)
#plt.plot(np.arange(N),estBmr,label= "Con MR")
#plt.plot(np.arange(N),estBZ,label= "Interferente con interpolacion de ceros")
#plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
#plt.grid()


# Le resto la interferente a la señal de ecg original
est_ecg = ecg_one_lead - estBmr


# Grafico el resultado
#plt.figure(6)
#plt.plot(np.arange(N),ecg_one_lead,label= "ECG original")
#plt.plot(np.arange(N),est_ecg,label= "ECG sin interferente")
#plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
#plt.grid()

# Grafico el resultado
#plt.figure()
#plt.plot(np.arange(N),est_ecg,label= "ECG sin interferente")
#for xc in posQRS:
#    plt.axvline(x=xc,ymin=0.05,ymax=0.95,color = 'b', linestyle='--')
#plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
#plt.grid()
'''

region_0 = np.arange(10000,100000)
region_1 = np.arange(620000,720000)
region_2 = np.arange(20000,30000)

#regAnalisis = est_ecg[10000:100000]
regAnalisis = est_ecg[region_2]


widths = np.arange(1,101)
cwtmatr, freqs = pywt.cwt(regAnalisis, widths, 'gaus1')


plt.figure()
plt.subplot(2,1,1)
# Para los colores de la escala ver: https://matplotlib.org/examples/color/colormaps_reference.html
plt.imshow(cwtmatr, extent=[0, len(regAnalisis), 1, 101], cmap='seismic', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())   
#plt.plot(np.arange(len(regAnalisis)),regAnalisis)
#plt.colorbar()

plt.subplot(2,1,2)
plt.plot(np.arange(len(regAnalisis)),regAnalisis,label= "ECG")
#plt.plot(np.arange(len(regAnalisis)),cwtmatr[2,:],label= "wt2")
plt.plot(np.arange(len(regAnalisis)),cwtmatr[4,:],label= "wt4")
plt.plot(np.arange(len(regAnalisis)),cwtmatr[8,:],label= "wt8")
#plt.plot(np.arange(len(regAnalisis)),cwtmatr[16,:],label= "wt16")
#plt.plot(np.arange(len(regAnalisis)),cwtmatr[64,:],label= "wt64")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()


# Detecto todos los latidos mediante la deteccion de las volores que sobrepasan un umbral
# Normalizo el resultado de la transformada wavelet
wt8, freqs = pywt.cwt(est_ecg, [8], 'gaus1')
wt8 = np.transpose(wt8)
wt8 = wt8/np.max(wt8)

# Constantes
TIME_OFF = 300      # Tiempo que no detecto niveles superiores al umbral despues de superar uno
cont = 0
flag = 0

R_TRH = 0.22
detectR = []
VENT_WIDTH = 40
cruceCero = 0

for i in range(len(wt8)):
    
    if flag != 1:
        # Detecto los pulsos que superan el umbral
        if wt8[i] >= R_TRH:   
            
            # Obtengo el minimo local dentro de una ventana a partir del valor que supera el umbral
            minLocal = np.argmin(wt8[i-VENT_WIDTH:i])
            
            # Detecto el cruce por cero
            cruceCero = np.argmin(np.abs(wt8[i-minLocal:i]))
            detectR.append(i-minLocal+cruceCero)
            
            flag = 1
    else:
        cont = cont+1
        if cont == TIME_OFF:
            cont = 0
            flag = 0

detectR = np.transpose(np.array(detectR))


# Hago la matriz de confucion 
# Genero un array del largo del ecg en el cual pongo en 1 el lugar donde hay un latido
refR = np.zeros(N)              # Posicion de los latidos del archivo 'qrs_detections'
detR = np.zeros(N)              # Posicion de los latidos detectados con mi algoritmo

for i in posQRS:
    refR[i] = 1
    
for i in detectR:
    detR[i] = 1
    
FP = 0      # Falso Positovo
FN = 0      # Falso Negativo
VP = 0      # Verdadero Positivo
dt = 15     # Margen dentro del cual concidero valido el latido detectado

for i in posQRS:
    if np.sum(detR[i-dt:i+dt]) == 1:
        VP = VP+1
    else:
        FN = FN+1
        
for i in detectR:
    if np.sum(refR[i-dt:i+dt]) == 0:
        FP = FP+1

acc = VP/len(posQRS)    # Exactitud
vpp = VP/(VP+FP)        # Valor predictivo positivo
S = VP/(VP+FN)          # Sensibilidad
    
print("Ventana de tiempo dt="+str(dt*2)+"ms")
print("Cantidad total de latidos: "+str(len(posQRS)))
print("Verdadero-Positovo (VP): "+str(VP))
print("Falso-Positivo (FP): "+str(FP))
print("Falso-Negativo (FN): "+str(FN))
print("Exactitud: %.2f" %(acc*100)+"%")
print("Valor predictivo positivo: %.2f" %(vpp*100)+"%")
print("Sensibilidad: %.2f" %(S*100)+"%")
# El FN y FP dan distinto de cero 
# pero encontre que falta un latido cerca de 247809 en los latidos detectados que asumo como verdaderos


# Grafico los latidos detectados con el algoritmo y los que estan en qrs_detections
region = np.arange(246500,249500)  # En esta region encontre diferencias

plt.figure()
plt.title("Diferencia entre la ubicación de los latidos")
plt.plot(region,est_ecg[region],label= "ECG")

indice = np.argwhere(detectR>region[0])
indFin = 3

plt.axvline(x=detectR[indice[0]],ymin=0.6,ymax=1,color = 'r', linestyle='--',label = "Detección con wavelet")
plt.axvline(x=posQRS[indice[0]],ymin=0,ymax=0.4,color = 'g', linestyle='--',label = "Ubicación qrs_detections")

for xc in detectR[indice[1:indFin+1]]:
    plt.axvline(x=xc,ymin=0.6,ymax=1,color = 'r', linestyle='--')
   
for xc in posQRS[indice[1:indFin]]:
    plt.axvline(x=xc,ymin=0,ymax=0.4,color = 'g', linestyle='--')

plt.ylim(-15000, 26000)
plt.xlabel("Timepo [s]")
plt.legend(bbox_to_anchor=(0.80, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()

# 0.2
# 40

# Clasifico cuales son latidos normales y cuales son ventriculares
VENTR_THR = 0.25         # Umbral para detectar latidos ventriculares
VENT_WIDTH = 70

pLatNorm = []
pLatVentr = []

wt128, freqs = pywt.cwt(est_ecg, [128], 'gaus1')
wt128 = np.transpose(wt128)
wt128 = wt128/np.max(wt128)

for r in detectR:
    
    if np.max(wt128[r:r+VENT_WIDTH]) > VENTR_THR:
        pLatVentr.append(r)
    else:
        pLatNorm.append(r)

pLatVentr = np.transpose(np.array(pLatVentr))
pLatNorm = np.transpose(np.array(pLatNorm))

'''
plt.figure()
plt.plot(np.arange(len(est_ecg)),est_ecg/np.max(est_ecg),label= "ECG")
plt.plot(np.arange(len(est_ecg)),wt128,label= "wt128")
plt.grid(True)
plt.show()
'''

'''
dtp = 350
dtn = 250
maxECG = np.max(est_ecg)
plt.figure()

plt.subplot(1,2,1)
plt.title("Latidos normales")
for norm in pLatNorm:
    plt.plot(np.arange(dtn+dtp),est_ecg[norm-dtn:norm+dtp]/maxECG,color = 'b')
plt.grid(True)
plt.ylim(-0.4,1.1)
  
plt.subplot(1,2,2)
plt.title("Latidos ventriculares")
for ventr in pLatVentr:
    plt.plot(np.arange(dtn+dtp),est_ecg[ventr-dtn:ventr+dtp]/maxECG,color = 'g')  
plt.ylim(-0.4,1.1)
plt.grid(True)
plt.show()
'''

'''
# Grafico la ubicacion de los latidos ventriculares y los normales
plt.figure()
plt.plot(np.arange(len(est_ecg)),est_ecg/np.max(est_ecg),label= "ECG")

for xc in pLatNorm:
    plt.axvline(x=xc,ymin=0.5,ymax=0.95,color = 'r', linestyle='--')
 
for xc in pLatVentr:
    plt.axvline(x=xc,ymin=0.05,ymax=0.5,color = 'g', linestyle='--')
    
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()
'''


# Calculo el segmento QT de los latidos normales
Q_THR = 0.05    # Umbral para detectar el inicio de la onda Q con respecto al modulo maximo de la trasformada wavelet en el latido en evaluacion
T_THR = 0.4    # Umbral para detectar el fin de la onta T

VENT_WIDTH_Q =  60
#VENT_WIDTH_T_FIN =  350
VENT_WIDTH_T_FIN =  300
VENT_WIDTH_T_INI =  100

wt4, freqs = pywt.cwt(est_ecg, [4], 'gaus1')
wt4 = np.transpose(wt4)
wt4 = wt4/np.max(wt4)

wt32, freqs = pywt.cwt(est_ecg, [32], 'gaus1')
wt32 = np.transpose(wt32)
wt32 = wt32/np.max(wt32)

# Determino el inicio de la onda Q 
inicioQ = []

for r in pLatNorm:
    
    # Detecto el minimo local
    minLoc = np.argmin(wt4[r-VENT_WIDTH_Q:r])
    wt4MaxMod = np.abs(wt4[r-VENT_WIDTH_Q+minLoc])
    
    for i in range(minLoc-1,-1,-1):
        if np.abs(wt4[r-VENT_WIDTH_Q+i]) <= wt4MaxMod*Q_THR:
            inicioQ.append(r-VENT_WIDTH_Q+i)
            break

inicioQ = np.transpose(np.array(inicioQ))

# Determino en fin de la onda T
finT = []           

for r in pLatNorm:
    
    # Detecto el maximo local perteneciente a la onda T
    maxLoc = np.argmax(wt32[r+VENT_WIDTH_T_INI:r+VENT_WIDTH_T_FIN])
    wt32MaxMod = np.abs(wt32[r+maxLoc])
    
    for i in range(VENT_WIDTH_T_FIN-VENT_WIDTH_T_INI):
        if np.abs(wt32[r+VENT_WIDTH_T_INI+maxLoc+i]) <= wt32MaxMod*T_THR:
            finT.append(r+VENT_WIDTH_T_INI+maxLoc+i)
            break

finT = np.transpose(np.array(finT))


plt.figure()
plt.plot(np.arange(len(est_ecg)),est_ecg/np.max(est_ecg),label= "ECG")
#plt.plot(np.arange(len(est_ecg)),wt4,label= "wt4")
#plt.plot(np.arange(len(est_ecg)),wt32,label= "wt32")

plt.axvline(x=pLatNorm[0],ymin=0.75,ymax=0.95,color = 'r', linestyle='--',label = "Latido normal")
plt.axvline(x=inicioQ[0],ymin=0.05,ymax=0.25,color = 'g', linestyle='--',label = "Inicio onta Q")
plt.axvline(x=finT[0],ymin=0.05,ymax=0.25,color = 'b', linestyle='--',label = "Fin onda T")

for xc in pLatNorm:
    plt.axvline(x=xc,ymin=0.75,ymax=1,color = 'r', linestyle='--')
 
for xc in inicioQ:
    plt.axvline(x=xc,ymin=0,ymax=0.35,color = 'g', linestyle='--')

for xc in finT:
    plt.axvline(x=xc,ymin=0,ymax=0.35,color = 'b', linestyle='--')
    
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()



# Obtengo el tacograma (curva de intervalos R-R)
intervaloRR = []

for i in range(len(detectR)-1):   
    intervaloRR.append(detectR[i+1]-detectR[i])
    
intervaloRR = np.transpose(np.array(intervaloRR))

# Filtro el tacograma
#tacograma = sig.sosfiltfilt(bpSosButter, intervaloRR)
#tacograma = sig.medfilt(intervaloRR,75)
#tacograma = sig.medfilt(intervaloRR,15)

tacogr = sig.medfilt(intervaloRR,9)
# Filtro de media movil
tacograma = []
ventana = 8
'''
for i in range(len(tacogr)-ventana):
    tacograma.append(np.mean(tacogr[i:i+ventana]))
tacograma = np.array(tacograma)
'''
tacograma = np.convolve(tacogr, np.ones((ventana,))/ventana, mode='valid')

#tacograma = sig.medfilt(tacograma,5)

plt.figure()
plt.plot(np.arange(len(intervaloRR)),intervaloRR,label= "R-R todo los latidos")
plt.plot(np.arange(len(tacogr)),tacogr,label= "tacogr")
plt.plot(np.arange(len(tacograma)),tacograma,label= "Tacograma")
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()



# Obtengo el grafico de intervalos QT
intRRnormal = []
intervaloQT = []
for i in range(len(pLatNorm)-1):   
    intRRnormal.append(pLatNorm[i+1]-pLatNorm[i])
    intervaloQT.append(finT[i]-inicioQ[i])

intRRnormal = np.transpose(np.array(intRRnormal))
intervaloQT = np.transpose(np.array(intervaloQT))

# Obtengo el grafico de segmento QT en funcion del intervalo RR

plt.figure()
#plt.plot(np.arange(len(intRRnormal)),intRRnormal,label= "RR")
plt.plot(np.arange(len(intervaloQT)),intervaloQT,label= "QT")

plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()


plt.figure()
plt.title('Scatter plot')
plt.scatter(intRRnormal, intervaloQT,label = "QT=f(RR)")
plt.axis([0, 2000, 0, 500])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.) 
plt.grid(True)
plt.show()


'''
intervaloQT = []
i=0
for r in detectR[:-1]:   
    
    # Si es un latido normal calculo el largo del segmento
    if r in pLatNorm.tolist():
        intervaloQT.append(finT[i]-inicioQ[i])
        i = i+1
    else:
        intervaloQT.append(finT[i-1]-inicioQ[i-1])
        
intervaloQT = np.transpose(np.array(intervaloQT))
'''
# Filtro el intervalo QT
# Filtro de mediana
interQT = sig.medfilt(intervaloQT,17)


intervaloQT2 = []
i=0
for r in detectR[:-1]:   
    
    # Si es un latido normal calculo el largo del segmento
    if r in pLatNorm.tolist():
        intervaloQT2.append(interQT[i])
        i = i+1
    else:
        intervaloQT2.append(interQT[i-1])
        
interQT = np.transpose(np.array(intervaloQT2))



# Filtro de media movil
intervaloQTfilt = []
ventana = 8
for i in range(len(interQT)-ventana):
    intervaloQTfilt.append(np.mean(interQT[i:i+ventana]))
intervaloQTfilt = np.array(intervaloQTfilt)

# Filtro de mediana
intervaloQTfilt = sig.medfilt(intervaloQTfilt,5)


plt.figure()
#plt.plot(np.arange(len(intRRnormal)),intRRnormal,label= "RR")
plt.plot(np.arange(len(intervaloQTfilt)),intervaloQTfilt,label= "QT filtrado")
plt.plot(np.arange(len(interQT)),interQT,label= "QT mediana 1")
plt.plot(np.arange(len(intervaloQT)),intervaloQT,label= "QT ")
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()




plt.figure()
plt.plot(np.arange(len(intervaloQTfilt)),intervaloQTfilt,label= "QT")
plt.plot(np.arange(len(tacograma)),tacograma,label= "Tacograma")
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)
plt.show()




plt.figure()
plt.title('Scatter plot')
plt.scatter(tacograma[680:1370], intervaloQTfilt[680:1370],label = "QT=f(RR) 1")
plt.scatter(tacograma[1400:-1], intervaloQTfilt[1400:-1],label = "QT=f(RR) 2")
plt.axis([0, 2000, 0, 500])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(0.98, 0.98), loc=2, borderaxespad=0.) 
plt.grid(True)
plt.show()


