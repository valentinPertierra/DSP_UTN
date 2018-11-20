# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:47:05 2018

@author: Valentin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import HTML

from funciones import *
from ventanas import *
from scipy.fftpack import fft
from scipy import signal

# Periodograma de Welch casero
def periodogramaWelch(x,K,overlap):
    
    # Determino el largo de la señal y la cantidad de realizaciones
    N = len(x[:,0])
    R = len(x[0,:])
    
    # Calculo el largo de cada bloque
    L = N/(1+(1-O)*(K-1))
    
    D = (1-O)*L 
    D = np.round(D)
    L = N-D*(K-1)
    
    D = int(D)
    L = int(L)
    
    # Inicializo vectores
    promPSDW = np.zeros(shape=(L//2,R))
    xw = np.zeros(shape=(L,R))
    
    # Promedio los PSD de los K bloques solapados
    for Ki in range(K):

        w = np.reshape(bartlett(L),(L,1))        
        xw = x[Ki*D:Ki*D+L,:]*w
               
        espW = fft(xw,axis=0)*(1/L)
        
        PSDW = np.abs(espW)**2
        PSDW = PSDW[:L//2]
        promPSDW = promPSDW+PSDW/K
        
    return promPSDW


fs = 1000 # Hz
N = 1000
df = fs/N

# Parametros para el metodo de Welch
K = 4       # Cantidad de bloques
O = 0.5     # Solapamiento

# Realizaciones
R = 200

# Parametros de la señal
a1 = np.sqrt(2)   # Esta normalizada en energia
f0 = fs/4

# Variable aleatoria con distribución uniforme para fr
va = np.random.uniform(-1/2,1/2,R)

# Genero un vector con el tiempo
t = np.linspace(0,(N-1)/fs,N)

# Genero la señal senoidal
x = []
for fr in va:
    f = f0+fr*df   
    x.append(a1*np.sin(t*2*np.pi*f))

x = np.transpose(np.array(x))

# Parametros del ruido normal
# El ruido tiene que estar 3 y 10db por abajo de la senoidal
u = 0                               # Media
db = -10
#v = (N/2)*10**(db/10)        # Varianza  
v = 40
# Genero señal de ruido
n = np.sqrt(v)*np.random.randn(N,R)+u

# Le sumo el ruido a la señal senoidal
x = x+n

# Obtengo el periodograma de Welch
MpsdWelch = periodogramaWelch(x,K,O)     #Este es el implementado por mi
psdWelch = np.zeros(shape=(201,R)) 
for i in range(R):
    f,psdWelch[:,i] = signal.welch(x[:,i], fs, 'bartlett',nperseg=400, noverlap=200)    # ver: scipy.signal.welch

plt.figure(1)
plt.plot(f,20*np.log10(psdWelch))

Mf = np.arange(len(MpsdWelch))*(fs/(len(MpsdWelch)*2))
plt.figure(2)
plt.plot(Mf,20*np.log10(MpsdWelch))

# Obtengo la frecuencia de la senoidal utilizando el estimador
estF0 = f[np.argmax(psdWelch,axis=0)]
valEspF0 = np.mean(estF0)
varEstF0 = np.var(estF0)

print(estF0)
print(valEspF0)
print(varEstF0)



'''
f = np.arange(len(psdWelch))*(fs/(len(psdWelch)*2))

# Obtengo la frecuencia de la senoidal utilizando el estimador
estF0 = f[np.argmax(psdWelch,axis=0)]

valEspF0 = np.mean(estF0)
varEstF0 = np.var(estF0)

print(estF0)
print(valEspF0)
print(varEstF0)

plt.figure(1)
plt.plot(f,20*np.log10(np.sum(psdWelch,axis=1)/R))
#plt.plot(f,20*np.log10(psdWelch))

plt.figure(2)
#plt.plot(f,20*np.log10(np.sum(psdWelch,axis=1)/R))
plt.plot(np.arange(len(estF0)),estF0)


# Determino los valores de fecuencias
f = np.arange(N//2)*df

#aplico la fft a la señal y la normalizo
esp = fft(x,axis=0)*(1/N)

#me quedo con la mitad de las muestras
esp = esp[:N//2]

plt.figure(3)
plt.plot(f,20*np.log10(np.abs(esp)))


'''

'''
resultados = []
tus_resultados = []


# Determino el largo de los bloques a promediar
L = Ni/(1+(1-O)*(K-1))

D = (1-O)*L 
D = np.round(D)
L = Ni-D*(K-1)

D = int(D)
L = int(L)

# Genero matriz con señales aleatoreas de ruido normal
x = np.sqrt(v)*np.random.randn(Ni,realizaciones)+u

promPSDW = np.zeros(shape=(L,realizaciones))
xw = np.zeros(shape=(L,realizaciones))

# Promedio los PSD de los K bloques de la señal
for Ki in range(K):
            
    # Aplicando una ventana al bloque
    w = np.reshape(bartlett(L),(L,1))        
    xw = x[Ki*D:Ki*D+L,:]*w
           
    espectroW = fft(xw,axis=0)*(1/L)
    
    PSDW = np.abs(espectroW)**2
    promPSDW = promPSDW+PSDW/K

# Resultados sin ventaneo
# Varianza del estimador
varPSD = np.var(promPSD,axis=1)*(L**2)
meanVarPSD = np.mean(varPSD)

# Valor esperado de la PSD
EPSD = np.mean(promPSD,axis=1)

# Calculo el sesgo
sesgo = v/Ni-np.mean(EPSD)
 
# Resultados aplicando la ventana:
# Varianza del estimador
varPSDW = np.var(promPSDW,axis=1)*(L**2)
meanVarPSDW = np.mean(varPSDW)

# Valor esperado de la PSD
EPSDW = np.mean(promPSDW,axis=1)

# Calculo el sesgo
sesgoW = v/Ni-np.mean(EPSDW)

resultados.append([sesgo,meanVarPSD,sesgoW,meanVarPSDW])
#print("N: "+str(Ni)+", Sesgo: "+str(sesgo)+", Varianza:"+str(meanVarPSD))


tus_resultados = [resultados[0],resultados[1],resultados[4],resultados[9],resultados[19],resultados[99]]

grafResultados = np.array(resultados)
N = np.arange(50,5050,50)

plt.figure(figsize=(10[0:2,10))
plt.subplot(211)
plt.title("Módulo del sesgo")
plt.plot(N,np.abs(grafResultados[:,0]),label="sin ventana")    
plt.plot(N,np.abs(grafResultados[:,2]),label="Bartlett")       
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)

plt.subplot(212)
plt.title("Varianza")
plt.plot(N,grafResultados[:,1],label="sin ventana")
plt.plot(N,grafResultados[:,3],label="Bartlett")
plt.axis([-10,N[-1]+10,0,1.5*v**2])
plt.xlabel('N [muestras]')
plt.legend(bbox_to_anchor=(0.90, 0.98), loc=2, borderaxespad=0.)
plt.grid(True)

'''