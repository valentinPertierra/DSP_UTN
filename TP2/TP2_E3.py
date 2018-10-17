# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:27:54 2018

@author: Valentin
"""

N = 1024
fs = 1000

realizaciones = 200

# Parametros de la señal
a0 = 2
p0 = 0
f0 = fs/4

# Variable aleatoria con distribución uniforme
variableAleatoria  = np.random.uniform(-2,2,realizaciones)

# Vectores donde guardo las distintas realizaciones
eDirichlet = []
eBartlett = []
eHann = []
eBlackman = []
eFlatTop = []

# Hago tantas realizaciones como variables aleatorias tenga
for fr in variableAleatoria:
    
    # Determino f1
    f1 = f0+fr*(fs/N)
    
    # Genero la señal senoidal
    tt,x = generador_senoidal(fs,f1,N,a0,p0)
    
    # Aplico las dstintas ventanas a la señal 
    wBartlett =x*bartlett(N)
    wHann = x*hann(N)
    wBlackman = x*blackman(N)
    wFlatTop = x*flatTop(N)

    # Obtengo el espectro de las señales
    f,ex = analizadorEspectro(x)
    f,eBa = analizadorEspectro(wBartlett)
    f,eH = analizadorEspectro(wHann)
    f,eBl = analizadorEspectro(wBlackman)
    f,eF = analizadorEspectro(wFlatTop)
    
    eDirichlet.append(ex)
    eBartlett.append(eBa)
    eHann.append(eH)
    eBlackman.append(eBl)
    eFlatTop.append(eF)
    

eDirichlet = np.array(eDirichlet)
eBartlett = np.array(eBartlett)
eHann = np.array(eHann)
eBlackman = np.array(eBlackman)
eFlatTop = np.array(eFlatTop)

'''
a0Dirichlet = np.abs(eDirichlet[:,N//4])
a0Bartlett = np.abs(eBartlett[:,N//4])
a0Hann = np.abs(eHann[:,N//4])
a0Blackman = np.abs(eBlackman[:,N//4])
a0FlatTop = np.abs(eFlatTop[:,N//4])
'''

'''
# Genero un vector con los valores del estimador de a0 para las distintas señales
a0Dirichlet = []
a0Bartlett = []
a0Hann = []
a0Blackman = []
a0FlatTop = []

for i in range(realizaciones):
    a0Dirichlet.append(np.abs(eDirichlet[i][N//4]))
    a0Bartlett.append(np.abs(eBartlett[i][N//4]))
    a0Hann.append(np.abs(eHann[i][N//4]))
    a0Blackman.append(np.abs(eBlackman[i][N//4]))
    a0FlatTop.append(np.abs(eFlatTop[i][N//4]))
  '''  
    
    
'''    
plt.figure(figsize=(16,22))
plt.subplot(511)
plt.title("Histograma de |$X^0_w(\Omega_0)$| utilizando el kernel de Dirichlet")
plt.hist(a0Dirichlet, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(512)
plt.title("Histograma de |$X^1_w(\Omega_0)$| utilizando la ventana Bartlett")
plt.hist(a0Bartlett, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(513)
plt.title("Histograma de |$X^2_w(\Omega_0)$| utilizando la ventana Hann")
plt.hist(a0Hann, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(514)
plt.title("Histograma de |$X^3_w(\Omega_0)$| utilizando la ventana Blackman")
plt.hist(a0Blackman, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(515)
plt.title("Histograma de |$X^4_w(\Omega_0)$| utilizando la ventana FlatTop")
plt.hist(a0FlatTop, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.show()

'''




# Ejercicio 3_b

a = N//4-2
b = N//4+3


a0Dirichlet = np.abs(eDirichlet[:,a:b]/2)**2
a0Bartlett = np.abs(eBartlett[:,a:b]/2)**2
a0Hann = np.abs(eHann[:,a:b]/2)**2
a0Blackman = np.abs(eBlackman[:,a:b]/2)**2
a0FlatTop = np.abs(eFlatTop[:,a:b]/2)**2

a0Dirichlet = np.sqrt(np.sum(a0Dirichlet, axis=1)*(1/5))
a0Bartlett = np.sqrt(np.sum(a0Bartlett, axis=1)*(1/5))
a0Hann = np.sqrt(np.sum(a0Hann, axis=1)*(1/5))
a0Blackman = np.sqrt(np.sum(a0Blackman, axis=1)*(1/5))
a0FlatTop = np.sqrt(np.sum(a0FlatTop, axis=1)*(1/5))

plt.figure(figsize=(16,22))
plt.subplot(511)
plt.title("Histograma de |$X^0_w(\Omega_0)$| utilizando el kernel de Dirichlet")
plt.hist(a0Dirichlet, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(512)
plt.title("Histograma de |$X^1_w(\Omega_0)$| utilizando la ventana Bartlett")
plt.hist(a0Bartlett, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(513)
plt.title("Histograma de |$X^2_w(\Omega_0)$| utilizando la ventana Hann")
plt.hist(a0Hann, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(514)
plt.title("Histograma de |$X^3_w(\Omega_0)$| utilizando la ventana Blackman")
plt.hist(a0Blackman, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.subplot(515)
plt.title("Histograma de |$X^4_w(\Omega_0)$| utilizando la ventana FlatTop")
plt.hist(a0FlatTop, 10)
plt.axis([0,2,0,100])          
plt.grid(True)

plt.show()