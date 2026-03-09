# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:38:01 2025

@author: magui

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

N = 1000     # Número de muestras
fs = N       # Frecuencia de muestreo
T = 1 / fs   # Tiempo de muestreo
df = fs / N  # Resolución espectral  
dc = 0       #Desplazamiento vertical [V]  
ph = 0       #FASE = Desplazamiento horizontal [rad]


# Definicion funciónes seno
def func_sen(Vmax=1, dc=dc, ff=1, ph=ph, nn=N, fs=fs):
    
    tt = np.arange(0, nn / fs, 1/fs ).reshape(nn,1) ## arange ( START, STOP, STEP)
    
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph).reshape(nn,1) + dc
    
    return tt, xx

tt1, xx1 = func_sen(ff=1, fs=fs)
tt2, xx2 = func_sen(ff=500, fs=fs)
tt3, xx3 = func_sen(ff=999, fs=fs)
tt4, xx4 = func_sen(ff=1001, fs=fs)
tt5, xx5 = func_sen(ff=2001, fs=fs)


# GRAFICO 
# Caracteristicas del Grafico

plt.figure()
plt.title("Señales Senoidales sobre eje tiempo ")
plt.plot(tt2, xx2, label="Frec = 500 Hz", color = 'g')
plt.plot(tt3, xx3, label="Frec = 999 Hz", color = 'y')
plt.plot(tt1, xx1, label="Frec = 1 Hz", color = 'b')
plt.plot(tt4, xx4, label="Frec = 1001 Hz", color = 'black', linestyle = '--')
plt.plot(tt5, xx5, label="Frec = 2001 Hz", color = 'orange', linestyle = 'dotted')

plt.legend(loc = 'lower left')
plt.xlabel("Tiempo [segundos]")
plt.ylabel("Amplitud [Volts]")
plt.grid(True)
plt.show()



#%% BONUS 

def func_square(Vmax=1, dc=dc, ph=ph, nn=N, fs=fs, ff=1, duty= 1/2):
    
    tt = np.arange(0, nn/fs, 1/fs).reshape(nn, 1)
    
    xx = Vmax * signal.square(2 * np.pi * tt * ff + ph, duty=duty ).reshape(nn, 1) + dc
    
    return tt, xx



def func_triang(Vmax=1, dc=dc, ph=ph, nn=N, ff=1, fs=fs, width=1):
    
    tt = np.arange(0, nn/fs, 1/fs).reshape(nn, 1)
    
    xx = Vmax * signal.sawtooth(2 * np.pi * ff * tt + ph, width =width).reshape(nn, 1) + dc
   
    return tt, xx



##GRAFICO 2

tta, xxa = func_square (ff=3)    # 3 periodos enteros de la cuadrada
ttb, xxb =func_triang(ff=3)
plt.figure()
plt.title("Señales BONUS")
plt.plot(tta, xxa, label="Señal cuadrada")
plt.plot(ttb, xxb, label=" Señal 'diente de sierra' ", linestyle = '--')
plt.legend(loc = 'lower left')
plt.xlabel("Tiempo [segundos]")
plt.ylabel("Amplitud [Volts]")
plt.grid(True)
plt.show()
