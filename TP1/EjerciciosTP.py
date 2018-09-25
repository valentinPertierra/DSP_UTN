# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:20:05 2018
@author: Valentin
-------------------------------------------------------------------------------
Ejercicios del TP1
"""
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from funciones import *
from senialesEjercicio4 import *
from pandas import DataFrame
from IPython.display import HTML

from time import time
import timeit


#%%
# Ejercicio 0
# Prueba de la funcion para generar señales senoidales
def ejercicio_0():
    # Genero señales senoidales:
    f1 = 10
    a1 = 1
    p1 = 0
    
    f2 = 20
    a2 = 0.5
    p2 = 0
    
    N = 512
    fs = 100
    
    # Parametro de las señales senoidales
    paramSeno = [[fs,f1,N,a1,p1],[fs,f2,N,a2,p2],[fs,f2+5,N,a2,p2]]    
    senoidal = []
    
    for i in range(len(paramSeno)):
        # Genero un vector con las señales senoidales
        t,y = generador_senoidal(paramSeno[i][0],paramSeno[i][1],paramSeno[i][2],paramSeno[i][3],paramSeno[i][4])
        senoidal.append(y)
        
    # Transpongo la matriz
    senoidal = np.transpose(senoidal)
    
    # Grafico
    graficar(t,senoidal,1,'Senoidal','f(x)','t')
    plt.show()

#%%
# Ejercicio 2_a
def ejercicio_2_a():
    
    N = 1000
    fs = 1000
    
    a0 = 1
    p0 = 0
    f0 = fs/100
    
    D = 0.5
    S = 0.5
    
    tts,senoidal = generador_senoidal(fs,f0,N,a0,p0)
    ttc,cuadrada = generador_cuadrada(fs, f0, N, a0, D)
    ttt,triangular = generador_triangular(fs, f0, N, a0*2, S)
    
    # Genero vector con fecuencias
    df = fs/N
    ff = np.arange(int(N/2))*df
    
    # Hago la DFT y normalizo el espectro
    DFTsenoidal = DFT(senoidal)*(2/N)
    DFTcuadrada = DFT(cuadrada)*(2/N)
    DFTtriangular = DFT(triangular)*(2/N)
    
    # Me quedo con la mitad de las muestras
    DFTsenoidal = DFTsenoidal[:int(N/2)]
    DFTcuadrada = DFTcuadrada[:int(N/2)]
    DFTtriangular = DFTtriangular[:int(N/2)]
    
    # Grafico las señales y el modulo del espectro
    plt.figure(figsize=(16,16))
    plt.subplot(321)
    plt.title("Señal senoidal con f0="+str(f0)+"Hz")
    plt.plot(tts, senoidal)           
    plt.xlabel('t [s]')
    plt.ylabel('|f(t)|')
    plt.grid(True)
    
    plt.subplot(322)
    plt.title("Espectro senoidal aplicando la DFT")
    plt.stem(ff, np.absolute(DFTsenoidal))           
    plt.xlabel('f [Hz]')
    plt.ylabel('|X(f)|')
    plt.axis([0,200,0,np.amax(np.absolute(DFTsenoidal))*1.1])
    plt.grid(True)
    
    plt.subplot(323)
    plt.title("Señal cuadrada con f0="+str(f0)+"Hz; D="+str(D))
    plt.plot(ttc, cuadrada)           
    plt.xlabel('t [s]')
    plt.ylabel('|f(t)|')
    plt.grid(True)
    
    plt.subplot(324)
    plt.title("Espectro cuadrada aplicando la DFT")
    plt.stem(ff, np.absolute(DFTcuadrada))           
    plt.xlabel('f [Hz]')
    plt.ylabel('|X(f)|')
    plt.axis([0,200,0,np.amax(np.absolute(DFTcuadrada))*1.1])
    plt.grid(True)
    
    plt.subplot(325)
    plt.title("Señal triangular con f0="+str(f0)+"Hz; S="+str(S))
    plt.plot(ttt, triangular)           
    plt.xlabel('t [s]')
    plt.ylabel('|f(t)|')
    plt.grid(True)
    
    plt.subplot(326)
    plt.title("Espectro triangular aplicando la DFT")
    plt.stem(ff, np.absolute(DFTtriangular))           
    plt.xlabel('f [Hz]')
    plt.ylabel('|X(f)|')
    plt.axis([0,200,0,np.amax(np.absolute(DFTtriangular))*1.1])
    plt.grid(True)
    
    plt.show()
    
#%%
# Ejercicio 2_b
def ejercicio_2_b():

    N = [16,32,64,128,256,512,1024,2048]
    fs = 1000
    
    # Parámetros de la señal
    a0 = 1
    p0 = 0
    f0 = fs/4
    
    # Vector con los los resultados
    resultados = [ ['--','--']]
    
    for n in N:
        
        tts,senoidal = generador_senoidal(fs,f0,n,a0,p0)
            
        tStartDFT = time()
        espectroDFT = DFT(senoidal)
        dtDFT = time()-tStartDFT
            
        tStartFFT = time()
        espectroFFT= fft(senoidal)
        dtFFT = time()-tStartFFT
        
        resultados.append([dtDFT,dtFFT])
    
    print(resultados)
    
    
    
#%%
# Ejercicio 3_a
def ejercicio_3_a():
    
    N = 1024
    a0 = 1
    p0 = 0
    
    fs = 1000
    f0 = fs/4
    
    fd = [0,0.01,0.25,0.5]
    
    
    for i in fd:
        
        f = (N/4+i)*fs/N
        tt,signal = generador_senoidal(fs,f,N,a0,p0)
        ff,espectro = analizadorEspectro(signal,tt)
        
        Xo = np.absolute(espectro[int(N/4)])            #|X(fo)|
        X1 = np.absolute(espectro[int(N/4)+1])          #|X(fo+1)|
        
        # Calculo la sumatoria de los modulos para todas las frecuencias
        Xj = 0
        for j in espectro:
            Xj = Xj+np.absolute(j)
            
        Xj = Xj-Xo      # A la sumatoria le resto |X(fo)|
        
        # Calculo la energia total
        et = 0
        for j in espectro:
            et = et+np.absolute(j)*np.absolute(j)
            
        
        print("Para Fd="+str(i)+":")
        print("|X(fo)|="+str(Xo)+"; |X(fo+1)|="+str(X1)+"; Sumatoria menos |X(fo)|="+str(Xj))
        print("Energia = "+str(et))
       
        #grafico el modulo y la fase del espectro
        graficarEspectro(ff,espectro,0,"Espectro Fd="+str(i))
           
    plt.show()

#%% Ejercicio 3.b
def ejercicio_3_b():
    N = 1024
    a0 = 1
    p0 = 0
    
    fs = 1000
    f0 = fs/4
    
    Mj = [int(N/10),N,10*N]
    
    
    for i in Mj:
        print(N+i)
        tt,signal = generador_senoidal(fs,f0,N+i,a0,p0)
        signal[N:]= 0                                       # Hago cero los valores desde N hasta el final de la señal
       
        ff,espectro = analizadorEspectro(signal,tt)
        
        #grafico el modulo y la fase del espectro
        graficarEspectro(ff,espectro,0,"Espectro Mj="+str(i))
           
    plt.show()

#%%  Ejercicio 4
def ejercicio_4():
    
    N = 1024
    fs = 1000

    f0 = 9*fs/N
    
    # Genero la señal (las señales etan en el archivo senalesEjercicio4.py)
    tt,signal = signal_4_b(fs,N)
    tt,signalC = signal_cuadrada(fs,N)
    
    # Calculo la energia de la señal en el tiempo
    energiaTiempo=0
    for x in signal:
        energiaTiempo += x**2
        
    energiaTiempo = energiaTiempo/N    
    
    # Obtengo el espectro
    ff,espectro = analizadorEspectro(signal,tt)
    ff,espectroC = analizadorEspectro(signalC,tt)
    
    # Calculo la energia en el espectro
    # Cuando hago la fft multiplico el espectro por 2/N para
    # escalarlo (porque me quedo con la mitad). Pero para calcular la energia
    # necesito elevar al cuadrado el modulo de todo el espectro. Por esto lo 
    # divido por 2 y hago la sumatoria de los cuadrados del modulo, de esta 
    # forma obtengo la mitad de la energia y al final la duplico
    energiaTotal = 0
    for x in espectro:
        energiaTotal += np.absolute(x/2)**2     
    
    energiaTotal = energiaTotal*2
    
    # Calculo la energia en |X(f0)|
    energiaF0 = (np.absolute(espectro[int((f0/fs)*N)]/2)**2)*2
    
    # Obtengo la frecuencia de la cmoponente con mayor energia
    moduloEspectro = np.absolute(espectro)
    binMaxAmplitud = np.argmax(moduloEspectro)
    
    # Grafico señal
    graficar(tt,signal,1,'Senoidal con $f_0=9*f_s/N$','f(x)','t')
    graficar(tt,signalC,1,'Senoidal con $f_0=9*f_s/N$','f(x)','t')
    
    # Grafico el espectro
    graficarEspectro(ff,espectro,2,"Espectro $f_0=9*f_s/N$")
    graficarEspectro(ff,espectroC,2,"Espectro $f_0=9*f_s/N$")
    
    print("Energia total (Tiempo) = "+str(energiaTiempo))
    print("Energia total (Espectro) = "+str(energiaTotal))
    print("Energia en |X(f0)| = "+str(energiaF0))
    print("Frecuencia que maximiza la energia = "+str(binMaxAmplitud*(fs/N))+"Hz"+"; "+str(binMaxAmplitud)+"*fs/N")
    

#%%
# Ejercicio 5
def ejercicio_5():
    N = 1024
    fs = 1000
    
    # Parametros de la senoidal
    a0 = np.sqrt(2)
    p0 = 0
    f0 = 10*(fs/N)
    
    # Parametros del cuantizador
    bits = 16
    cuentas = 2**(bits-1)-1
    q=2/(2**bits)
    
    # Parametros del ruido
    u = 0       # Media
    v = 0.1    # Varianza
    
    
    # Genero señal senoidal 
    tt,signal = generador_senoidal(fs,f0,N,a0,p0)
    
    tt,ruido = genRuidoNormal(u,v,N,fs)
    ruido = np.transpose(ruido)
    ruido = np.reshape(ruido,N)
    
    # Sumo el ruido
    signal = signal + ruido
    
    #normalizo la señal respecto del valor máximo
    signal = signal/np.amax(np.abs(signal))
    graficar(tt,signal,1,'Senoidal con ruido','f(x)','t')
    
    # Cuantizo la señal
    signalQ = cuantizador(signal,bits,"ROUND")
    graficar(tt,signalQ,2,'Senoidal cuantizada','f(x)','t')
    
    # Obtengo el error
    error = signalQ-signal*cuentas
    
    # grafico el error 
    graficar(tt,error,3,'error','f(x)','t')
    
    # Hago la fft del error
    ff,espectroE = analizadorEspectro(error,tt)
        
    #grafico el espectro del error
    graficarEspectro_dB(ff,espectroE,4,"Espectro error")
    
    #grafico el espectro de la señal cuantizada
    ff,espectroQ = analizadorEspectro(signalQ/cuentas,tt)
    graficarEspectro_dB(ff,espectroQ,5,"Espectro señal cuantizada")
    
    #grafico el espectro de la señal sin cuantizar
    ff,espectro = analizadorEspectro(signal,tt)
    graficarEspectro_dB(ff,espectro,6,"Espectro señal sin cuantizar")
    
    # grafico el histograma del error
    histograma(error,10)
    
    
    
    plt.show()

#%%
# Ejercicio 5
def ejercicio_5_a():
    N = 1024
    fs = 1000
    
    # Parametros de la senoidal
    a0 = np.sqrt(2)     # Senoidal con energía uitaria
    p0 = 0
    f0 = 10*(fs/N)
    
    # Parametros del cuantizador
    bits = [4,8,16]
    
    # Parametros del ruido
    u = 0       # Media
    v = 0.01    # Varianza
    
    # Vector en donde guardar los resultaos [Energía total,Energía totalQ, Energía total e]
    tus_resultados = [ ['$\sum_{f=0}^{f_S/2} \lvert S_R(f) \rvert ^2$', '$\sum_{f=0}^{f_S/2} \lvert S_Q(f) \rvert ^2$', '$\sum_{f=0}^{f_S/2} \lvert e(f) \rvert ^2$' ], 
                    ['','','']]
    
    # Genero señal senoidal 
    tt,signal = generador_senoidal(fs,f0,N,a0,p0)
    
    # Genero ruido
    tt,ruido = genRuidoNormal(u,v,N,fs)
    
    # Sumo el ruido
    signal = signal + ruido
    
    # Calculo la energía total
    energiaTotal = energiaTiempo(signal)
    #print(energiaTotal)
    
    # Normalizo la señal respecto del valor máximo para no pasarme del rango del cuantizador
    # Esto hace que la energía total no sea unitaria pero se mantiene la relación entre la 
    # energía de la senoidal y el ruido
    maxS = np.amax(np.abs(signal))
    signal = signal/maxS
    
    # Obtengo el espectro de la señal
    ff,signalE = analizadorEspectro(signal,tt)
    
    # Calculo la energía total despues de la normalización
    energiaTotal = energiaFrecuencia(signalE)
    #print(energiaTotal)
    
    error = []
    errorE = []
    signalQ = []
    signalQE = []
    energiaTotalQ = 0
    energiaTotale = 0
    q = 0
    
    for b in bits:
        
        # Cuentas del cuantizador
        cuentas = 2**(b-1)-1
        q = 2/(2**b)
        
        # Cuantizo la señal
        sQ = cuantizador(signal,b,"CEIL")
        signalQ.append(sQ/cuentas)
        
        # Obtengo el espectro de la señal cuantizada
        ff,sQE = analizadorEspectro(sQ/cuentas,tt)
        signalQE.append(sQE)
        
        # Obtengo el error
        e = sQ/cuentas-signal
        #e = sQ-signal*cuentas
        error.append(e)
        
        # Obtengo el espectro del error
        ff,eE = analizadorEspectro(e,tt)
        errorE.append(eE)
        
        # Calculo la energía total Q
        energiaTotalQ = energiaFrecuencia(sQE)
        
        # Calculo la energía total e
        energiaTotale = energiaFrecuencia(eE)
                
        # Guardo los valores de energía en el vector de resultados
        tus_resultados.append(["%.5f"%energiaTotal,"%.5f"%energiaTotalQ,"%.5f"%energiaTotale])
    
    
    print(tus_resultados)
    
    # grafico el histograma del error
    histograma(error[0],10)
    histograma(error[1],10)
    histograma(error[2],10)
    
    # Obtengo el valor medio, el valor RMS y la energía total del error
    valorMedio = []
    valorRMS = []
    energiaT = []
    for i in range(len(error)):
        valorMedio.append(getValorMedio(error[i]))
        valorRMS.append(getValorRMS(errorE[i]))
        energiaT.append(energiaFrecuencia(errorE[i]))
        
    print("Valor medio = "+str(valorMedio))
    print("Valor RMS = "+str(valorRMS))
    print("Energía total = "+str(energiaT))
    
    # Obtengo el valor esperado, el desvio estandar y la varianza del error
    valorEsperado = []
    desvioEstandar = []
    varianza = []
    for i in range(len(error)):
        vE = getValorEsperado(error[i])
        valorEsperado.append(vE)
        dE = getDesvioEstandar(error[i])
        desvioEstandar.append(dE)
        varianza.append(dE**2)
        
    print("Valor eperado = "+str(valorEsperado))
    print("Desvio estandar = "+str(desvioEstandar))
    print("Varianza = "+str(varianza))
    
    plt.show()
#%%
# Funcion iterativa para ontener señales con energia normalizada
def energiaNormalizada():
    
    N = 1024
    fs = 1000

    f0 = 9*fs/N
    
    
    amplitud = np.linspace(2,4,100)
    
    for a0 in amplitud:
        # Genero la señal
        tt,signal = signal_4_a(fs,N,a0)
        
        # Calculo la energia de la señal en el tiempo
        energiaTiempo=0
        for x in signal:
            energiaTiempo += x**2
            
        energiaTiempo = energiaTiempo/N
        
        print("a0="+str(a0)+"-> E="+str(energiaTiempo))
       


#%%  
#ejercicio_3_b()    
#ejercicio_4()
#ejercicio_2_b()
ejercicio_5_a()




