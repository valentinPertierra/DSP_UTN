# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:08:43 2018

@author: Valentin
"""
# Importo los modulos que utilizo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft

#%%
""" 
Ventana de Bartlett
"""
def bartlett(L):
    
    N = L-1
    n = np.arange(0,N/2)
    
    W = 2*n/N
    
    if L%2:  #L es impar
        W = np.concatenate((W,2-2*np.arange((N+1)/2,(N+1))/(N+1)), axis=None)
    else:
        W = np.concatenate((W,W[::-1]), axis=None)

    return W

#%%
""" 
Ventana Hann
"""
def hann(N):
    
    if N%2:  #N es impar
        M = (N+1)/2
        n = np.arange(0,M)
        W = 0.5-0.5*np.cos(2*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[-2::-1]), axis=None)
    else:
        M = N/2
        n = np.arange(0,M)
        W = 0.5-0.5*np.cos(2*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[::-1]), axis=None)

    return W 
        
#%%
""" 
Ventana Blackman
"""
def blackman(N):
    
    
    if N%2:  #N es impar
        M = (N+1)/2
        n = np.arange(0,M)
        W = 0.42-0.5*np.cos(2*np.pi*n/(N-1))+0.08*np.cos(4*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[-2::-1]), axis=None)
    else:
        M = N/2
        n = np.arange(0,M)
        W = 0.42-0.5*np.cos(2*np.pi*n/(N-1))+0.08*np.cos(4*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[::-1]), axis=None)

    return W 
        
#%%
""" 
Ventana Flat-top
"""
def flatTop(N):
    
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
    
    if N%2:  #N es impar
        M = (N+1)/2
        n = np.arange(0,M)
        W = a0-a1*np.cos(2*np.pi*n/(N-1))+a2*np.cos(4*np.pi*n/(N-1))-a3*np.cos(6*np.pi*n/(N-1))+a4*np.cos(8*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[-2::-1]), axis=None)
    else:
        M = N/2
        n = np.arange(0,M)
        W = a0-a1*np.cos(2*np.pi*n/(N-1))+a2*np.cos(4*np.pi*n/(N-1))-a3*np.cos(6*np.pi*n/(N-1))+a4*np.cos(8*np.pi*n/(N-1))
        
        W = np.concatenate((W,W[::-1]), axis=None)

    return W 


