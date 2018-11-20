# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:29:01 2018

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

from scipy.signal import correlate

def acorrBiased(y):
  """Obtain the biased autocorrelation and its lags
  """
  r = correlate(y, y) / len(y)
  l = np.arange(-(len(y)-1), len(y))
  return r,l

# This is a port of the code accompanying Stoica & Moses' "Spectral Analysis of
# Signals" (Pearson, 2005): http://www2.ece.ohio-state.edu/~randy/SAtext/
def blackmanTukey(y, w, Nfft, fs=1):
  """Evaluate the Blackman-Tukey spectral estimator

  Parameters
  ----------
  y : array_like
      Data
  w : array_like
      Window, of length <= y's
  Nfft : int
      Desired length of the returned power spectral density estimate. Specifies
      the FFT-length.
  fs : number, optional
      Sample rate of y, in samples per second. Used only to scale the returned
      vector of frequencies.

  Returns
  -------
  phi : array
      Power spectral density estimate. Contains ceil(Nfft/2) samples.
  f : array
      Vector of frequencies corresponding to phi.

  References
  ----------
  P. Stoica and R. Moses, *Spectral Analysis of Signals* (Pearson, 2005),
  section 2.5.1. See http://www2.ece.ohio-state.edu/~randy/SAtext/ for original
  Matlab code. See http://user.it.uu.se/~ps/SAS-new.pdf for book contents.
  """
  M = len(w)
  N = len(y)
  if M>N:
    raise ValueError('Window cannot be longer than data')
  r, lags = acorrBiased(y)
  r = r[np.logical_and(lags >= 0, lags < M)]
  rw = r * w
  phi = 2 * fft(rw, Nfft).real - rw[0];
  f = np.arange(Nfft) / Nfft;
  return (phi[f < 0.5], f[f < 0.5] * fs)




fs = 1000 # Hz
N = 1000
df = fs/N

# Parametros para el metodo de Welch
K = 4       # Cantidad de bloques
O = 0.5     # Solapamiento

# Realizaciones
R = 1

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
db = -30
v = (N/2)*10**(db/10)        # Varianza  

# Genero señal de ruido
n = np.sqrt(v)*np.random.randn(N,R)+u

# Le sumo el ruido a la señal senoidal
x = x+n

btWin = signal.hamming(N)
psdBT, fBT = blackmanTukey(x, btWin, N, fs)





plt.figure(1)
plt.plot(fBT,psdBT)














'''
# Obtengo la autocorrelacion de la señal
Sxx = np.correlate(x[:,0], x[:,0], mode='full')
#Sxx = np.correlate(x[:,0], x[:,0], mode='same')
Sxx = np.reshape(Sxx,(len(Sxx),1))

# Le aplico una ventana a la secuencia de autocorrelacion
w = np.reshape(bartlett(len(Sxx)),(len(Sxx),1))
SxxW = Sxx*w

# Obtengo el periodograma de la autocorrelacion
psd = fft(SxxW)/len(SxxW)


plt.figure(1)
plt.plot(np.arange(len(Sxx)),Sxx)
plt.plot(np.arange(len(SxxW)),SxxW)

plt.figure(2)
plt.plot(np.arange(len(psd)),np.abs(psd)**2)

'''



