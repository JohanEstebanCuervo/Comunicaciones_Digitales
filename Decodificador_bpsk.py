# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:09:14 2022

@author: pci
"""
import numpy as np
import matplotlib.pyplot as plt

def upsample(signal,N):

  y = np.zeros(len(signal)*N)

  for i in range(len(signal)):

    y[i*N] = signal[i]

  return y

Noise= 3
muestras_simbolo= 5
tb = 1 # tiempo de bit
Fp = 3.6
ps = np.concatenate((np.zeros(muestras_simbolo),np.ones(muestras_simbolo),np.zeros(muestras_simbolo)))
ts = np.linspace(-1,1+(muestras_simbolo-1)/muestras_simbolo,muestras_simbolo*3)


Nbits = 5 #numero de bits
bits = np.random.randint(0,2,Nbits) #Señal binaria de 6 bits
bits_sample = upsample(bits, muestras_simbolo)
t_s = np.linspace(1,Nbits+(muestras_simbolo-1)/muestras_simbolo,Nbits*muestras_simbolo)
t_b = range(1,len(bits)+1)

s_ps = np.convolve(bits_sample,ps)[int(muestras_simbolo):-int(muestras_simbolo*2-1)]
s_ps[np.where(s_ps==0)]=-1

t = np.linspace(0,tb*Nbits,len(s_ps))
Portadora = np.cos(2*np.pi*Fp*t)
plt.stem(t,s_ps)
plt.show()
plt.plot(t,Portadora)
plt.show()


signal = s_ps*Portadora+np.random.random(len(s_ps))*Noise-Noise/2

plt.plot(t,signal)

####### decodificacion ############

Y1 = signal*Portadora

plt.plot(t,Y1)


decobits = np.mean(np.reshape(Y1, (-1,muestras_simbolo)),axis=1)*2
bits_re = np.zeros_like(decobits).astype(int)
bits_re[np.where(decobits>0)]=1

