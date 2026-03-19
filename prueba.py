# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:24:37 2026

@author: magui
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


N = 1000    # Número de muestras
fs = N       # Frecuencia de muestreo
T = 1 / fs   # Tiempo de muestreo
df = fs / N  # Resolución espectral  
##dc = 0       #Desplazamiento vertical [V], funciona como valor medio 
##ph = 0       #FASE = Desplazamiento horizontal [rad] ya lo define en la func como 0 pq no es muy usado
## por defecto ya lo define como 0, lo mismo que Vmax y valor medio (dc)


# Definicion funciónes seno
def func_sen( nn=N, fs=fs, Vmax=1, dc= 0, ff=1, ph= 0):
    
    tt = np.arange(0, nn / fs, T ).reshape(nn,1) ## arange ( START, STOP, STEP), T = 1/fs
    
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph).reshape(nn,1) + dc
    
    return tt, xx

tt1, xx1 = func_sen(ff=1, fs=fs)

plt.figure()
plt.plot(xx1)