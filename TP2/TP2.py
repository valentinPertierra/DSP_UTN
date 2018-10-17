# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:59:10 2018

@author: Valentin
"""
'''
TP2: Ventanas 

'''
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from funciones import *
from scipy import signal
from ventanas import *

#%% Ejercicio 2_a
'''
# Genero señales senoidales:
N = 1024
fs = 1000

# Parametros de la senoidal x1(k)
a01 = 1
p01 = 0
f01 = fs/4

# Parametros de la senoidal x2(k)
db = 40
a02 = a01/10**(db/20)
p02 = 0
f02 = f01+10*fs/N

# Genero las senoidales
tt1,x1 = generador_senoidal(fs,f01,N,a01,p01)
tt2,x2 = generador_senoidal(fs,f02,N,a02,p02)

x = x1+x2

# Grafico señal
#graficar(tt1,x,0,'x = x1+ x2','f(x)','t')

# Obtengo el espectro
ff,espectro = analizadorEspectro(x)

# Grafico el espectro
graficarEspectro_dB(ff,espectro,0,"Espectro $X(k)$")





# Parametros de la senoidal x2(k)
db_1 = -40
db_2 = -40
a02_1 = a01*10**(db_1/20)
a02_2 = a01*10**(db_2/20)

# La cuantización en con 16btis
b = 16
cuentas = 2**(b-1)-1

# Vuelvo a generar la señal x2(k)
tt2,x2_1 = generador_senoidal(fs,f02,N,a02_1,p02)
tt2,x2_2 = generador_senoidal(fs,f02,N,a02_2,p02)

signalx_1 = x1+x2_1
signalx_2 = x1+x2_2

# Normalizo las señales
signalx_1 = signalx_1/np.amax(np.abs(signalx_1))
signalx_2 = signalx_2/np.amax(np.abs(signalx_2))



# Cuantizo la señal con 16bits
signalx_1Q = cuantizador(signalx_1,b,"ROUND")/cuentas
signalx_2Q = cuantizador(signalx_2,b,"ROUND")/cuentas


graficar(tt2,signalx_1Q,0,'x = x1+ x2','f(x)','t')

# Obtengo el espectro
f,eSignalx_1Q = analizadorEspectro(signalx_1Q)
f,eSignalx_2Q = analizadorEspectro(signalx_2Q)

graficarEspectro_dB(f,eSignalx_1Q,0,"Espectro $X(k)$")
graficarEspectro_dB(f,eSignalx_2Q,0,"Espectro $X(k)$")

'''
#%% Ejercicio 2_c

# Genero señales senoidales:
N = 513
fs = 1000

# Parametros de la senoidal x1(k)
d1 = 0.001

a01 = 1
p01 = 0
f01 = fs/4+d1*(fs/N)


# Parametros de la senoidal x2(k)
db = -30
a02 = a01*10**(db/20)
p02 = 0
f02 = fs/4+10*fs/N

# Parametros del cuantizador
bits = 16

# Genero las senoidales
tt1,x1 = generador_senoidal(fs,f01,N,a01,p01)
tt2,x2 = generador_senoidal(fs,f02,N,a02,p02)

x = x1+x2

# Agrego cuantizador
xQ = cuantizador(x,bits,"ROUND")
xQ = xQ/(2**(bits-1)-1)                 #Normalizo la cuatizacion
# Grafico señal
#graficar(tt1,xQ,0,'x = x1+ x2','f(x)','t')

# Obtengo el espectro
ff,espectroX = analizadorEspectro(xQ)

# Grafico el espectro
graficarEspectro_dB(ff,espectroX,1,"Espectro $X(k)$")

# Genero ventaneando la señal
ventana_blackmanharris = signal.blackmanharris(len(xQ))
my_bartlett = flatTop(len(xQ))
np_bartlett = signal.flattop(len(xQ))

xQ_blackmanharris = xQ*ventana_blackmanharris
graficar(tt1,my_bartlett,0,'xQ_blackmanharris','x(t)','t')
graficar(tt1,np_bartlett,0,'xQ_blackmanharris','x(t)','t')

# Grafico la señal ventaneada
graficar(tt1,xQ_blackmanharris,0,'xQ_blackmanharris','x(t)','t')

# Obtengo el espectro de la señal ventaneada
ff,espectroX_blackmanharris = analizadorEspectro(xQ_blackmanharris)

# Grafico el espectro
graficarEspectro_dB(ff,espectroX_blackmanharris,1,"Espectro $X(k)$")




