# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:37:57 2022

@author: pci
"""

import numpy as np
import matplotlib.pyplot as plt

def upsample(signal,N):

  y = np.zeros(len(signal)*N)

  for i in range(len(signal)):

    y[i*N] = signal[i]

  return y

def Decode_PSK(signal,Fp,Fs,Tb): ##  Fp -> Frecuencia Portadora. Fs -> Frecuencia de muestreo. Tb -> Tiempo de Baudio
                                  ##  Fs*Tb debe ser un numero entero  primera restricción

    t = np.linspace(0, len(signal)/Fs,len(signal))
    Portadorai = np.cos(2*np.pi*Fp*t)
    Portadoraq = np.cos(2*np.pi*Fp*t+np.pi/2)
    
    Y1 = signal*Portadorai
    Y2 = signal*Portadoraq
    
    decobits1 = np.mean(np.reshape(Y1, (-1,int(Fs*Tb))),axis=1)*2
    decobits2 = np.mean(np.reshape(Y2, (-1,int(Fs*Tb))),axis=1)*-2
    
    return np.reshape(np.concatenate((decobits1,decobits2)),(2,-1)).T
    
    
    
Noise= 0
muestras_simbolo= 60
tb = 1 # tiempo de bit
Fp = 4
ps = np.concatenate((np.zeros(muestras_simbolo),np.ones(muestras_simbolo),np.zeros(muestras_simbolo)))
ts = np.linspace(-1,1+(muestras_simbolo-1)/muestras_simbolo,muestras_simbolo*3)


Nbits = 3 #numero de bits

bits1 = np.random.randint(0,2,Nbits) #Señal binaria de 6 bits
bits_sample1 = upsample(bits1, muestras_simbolo)

bits2 = np.random.randint(0,2,Nbits) #Señal binaria de 6 bits
bits_sample2 = upsample(bits2, muestras_simbolo)

t_s = np.linspace(1,Nbits+(muestras_simbolo-1)/muestras_simbolo,Nbits*muestras_simbolo)
t_b = range(1,len(bits1)+1)

s_ps1 = np.convolve(bits_sample1,ps)[int(muestras_simbolo):-int(muestras_simbolo*2-1)]
s_ps1[np.where(s_ps1==0)]=-1

s_ps2 = np.convolve(bits_sample2,ps)[int(muestras_simbolo):-int(muestras_simbolo*2-1)]
s_ps2[np.where(s_ps2==0)]=-1

t = np.linspace(0,tb*Nbits,len(s_ps1))
Portadorai = np.cos(2*np.pi*Fp*t)
Portadoraq = np.cos(2*np.pi*Fp*t+np.pi/2)

plt.stem(t,s_ps1)
plt.show()
plt.stem(t,s_ps2)
plt.show()
plt.plot(t,Portadorai)
plt.show()
plt.plot(t,Portadoraq)
plt.show()

ruido=np.random.random(len(s_ps1))*Noise-Noise/2

signal = s_ps1*Portadorai-s_ps2*Portadoraq+ ruido

pows = (signal@signal)/len(signal)
powr = (ruido@ruido)/len(ruido)
SNR = 20*(np.log10(pows)-np.log10(powr))

print('SNR = '+str(SNR)+'db')
plt.plot(t,signal)
plt.show()
####### decodificacion ############
desfase= 0
Portadorai = np.cos(2*np.pi*Fp*t+desfase)
Portadoraq = np.cos(2*np.pi*Fp*t+np.pi/2+desfase)

Y1 = signal*Portadorai

plt.plot(t,Y1)


decobits1 = np.mean(np.reshape(Y1, (-1,muestras_simbolo)),axis=1)*2
bits_re1 = np.zeros_like(decobits1).astype(int)
bits_re1[np.where(decobits1>0)]=1

Y2 = signal*Portadoraq

plt.plot(t,Y2)


decobits2 = np.mean(np.reshape(Y2, (-1,muestras_simbolo)),axis=1)*-2
bits_re2 = np.zeros_like(decobits2).astype(int)
bits_re2[np.where(decobits2>0)]=1
plt.show()


bits1[np.where(bits1==0)]=-1
bits2[np.where(bits2==0)]=-1

plt.scatter(decobits1,decobits2)
plt.scatter(bits1,bits2)


puntos = Decode_PSK(signal, Fp, muestras_simbolo/tb, tb)



