# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:45:56 2026

@author: magui
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


N = 64000   # Número de muestras
fs = N   # Frecuencia de muestreo
T = 1 / fs  # Tiempo de muestreo = Ts 
df = fs / N  # Resolución espectral  
dc = 0       #Desplazamiento vertical [V]  
ph = 0       #FASE = Desplazamiento horizontal [rad]




# Definicion funciónes seno
def func_sen(Vmax=1, dc=dc, ff=1, ph=ph, nn=N, fs=fs):
    
    N= np.arange(nn)
    tt = N/fs    
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph).reshape(nn,1) + dc
    
    return tt, xx

  

# Definicion funcion cuadrada
def func_square(Vmax=1, dc=dc, ph=ph, nn=N, fs=fs, ff=1, duty= 1/2):
   
    N= np.arange(nn)
    tta = N/fs
    xxa = Vmax * signal.square(2 * np.pi * tta * ff + ph, duty=duty ).reshape(nn, 1) + dc
    
    return tta, xxa

def func_escalon(Vmax=1, ff=1, fs=fs, nn= N, duracion = 1, comienzo = 0 ):
    N = np.arange(nn)
    t = N/fs
    u = np.where(t < duracion, Vmax, 0)
    return t, u
 
# Definicion funcion RTA AL IMPULSO

# def rtaimp(x, sist):
#     x = x.flatten()
#     N = len(x)
#     M = N//50
#     impulso = np.zeros(M)
#     impulso[0] = 1 
#     deltas = sist(impulso)
#     h = np.convolve(x, deltas, mode = 'valid')
#     R = np.arange(N-M +1)
#     th = R * T
    
#     return h, th

#%% EJERCICIO 1: 
    
A1 = 1
A2 = A1 * 10**(3/20)    
    
t1, y1 = func_sen(Vmax = 1, ff = 2000)                             ## SENOIDAL BASE 
t2, y2 = func_sen(ff = 2000, Vmax = A2, ph = np.pi/2)     ## AMPLIADA Y DESFAZADA
ta, ya = func_sen(ff = 1000)                             ## MITAD DE FREC
y3 = y1 * ya                                             ## MODELADA t3 = t1
#y4 = np.max(y3) * 0.75
max= np.max(y3)
min = np.min(y3)
y4 = np.clip(y3, a_min=min*0.75, a_max= max*0.75)
t5, u = func_escalon(duracion = 0.01)
t6, y6 = func_square(ff=4000)


## CALCULO DE ENERGIA / POTENCIA |

E1 = np.mean(y1**2)
E2 = np.mean(y2**2)
E3 = np.mean(y3**2)
E4 = np.mean(y4**2)
E5 = np.sum(np.abs(u)**2) 
E6 = np.sum(np.abs(y6)**2) * 1/N

#%% EJE 1 GRAFS:
    
# # GRAFICO 1:
     
senales = [
    ("Frecuencia = 2KHz", f"Muestras:{N}, Tiempo entre muestras:{T}, Potencia:{E1: .3f} ","OBS:solo se grafican 150 muestras", t1[:150], y1[:150], 'b'),
    ("Frecuencia = 2kHz, ph=π/2, Vmax = 2", f"Muestras:{N}, Tiempo entre muestras:{T}, Potencia:{E2: .3f} ", "OBS:solo se grafican 150 muestras",t2[:150], y2[:150], 'g'),
    ("Senoidal modelada",  f"Muestras:{N}, Tiempo entre muestras:{T}, Potencia:{E3: .3f} ","OBS:solo se grafican 150 muestras", t1[:150], y3[:150], 'y'),
    ("Senoidal al 75% de amplitud", f"Muestras:{N}, Tiempo entre muestras:{T}, Potencia:{E4: .3f}","OBS:solo se grafican 150 muestras", t1[:150], y4[:150], 'orange'),
]
                                       

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
axs = axs.flatten()
fig.subplots_adjust(hspace=0.4, wspace=0.3)


for i, (titulo, subtitulo, obs, tiempo, señal, color) in enumerate(senales):
    axs[i].plot(tiempo, señal, color=color)
    axs[i].set_title(f"{titulo} \n {subtitulo} \n{obs}", fontsize= 10)
    axs[i].set( xlabel="Tiempo [s]", ylabel="Amplitud [V]")
    axs[i].tick_params(axis='both', labelsize=7)  # achica nros de los ejes
    axs[i].xaxis.label.set_size(7)                # achica nombre eje X
    axs[i].yaxis.label.set_size(7)                # achica nombre eje Y
    axs[i].grid(True) 

# GRAFICO 2:

    
senales2 = [
    ("Señal Pulso Rectangular: 10ms ",f"Muestras:{N} Tiempo entre muestras:{T}, Potencia:{E5: .3f}"," OBS:solo se grafican 2000 muestras", t5[:2000], u[:2000], 'blue'),
    ("Señal Cuadrada ff= 4kHz ",f"Muestras:{N}, Tiempo entre muestras:{T}, Potencia:{E6}", " OBS:solo se grafican 300 muestras", t6[:300], y6[:300], 'violet')
    ]
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
axs = axs.flatten()
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, (titulo, subtitulo, obs, tiempo, señal, color) in enumerate(senales2):
    axs[i].plot(tiempo, señal, color=color)
    axs[i].set_title(f"{titulo} \n {subtitulo} \n{obs}", fontsize= 10)
    axs[i].set( xlabel="Tiempo [s]", ylabel="Amplitud [V]")
    axs[i].tick_params(axis='both', labelsize=7)  # achica nros de los ejes
    axs[i].xaxis.label.set_size(7)                # achica nombre eje X
    axs[i].yaxis.label.set_size(7)                # achica nombre eje Y
    axs[i].grid(True) 
    
plt.show()


    
#%% EJERCICIO 2:
    
    
import numpy as np
import matplotlib.pyplot as plt

# Definimos el rango de n para la entrada
n = np.arange(-5, 6)

# Función escalón
u = lambda x: np.heaviside(x, 1)


W0 = 1
Ts = 1

# Entrada: x[n] = u[n+1] - u[n-2]
x1 = u(n+1) - u(n-2)
x2 = (1/2)**n * u(n)
x3 = np.cos(W0 * n * Ts)

# Definimos el rango de n para h[n]
m = np.arange(-6, 20)

# Delta discreta
def delta(k):
    return np.array([1 if i == 0 else 0 for i in k])

# Respuesta impulsiva: h[n] = δ[n] - δ[n-4]
h = delta(m) - delta(m-4)

# Convolución: y[n] = x[n] * h[n]
y1 = np.convolve(x1, h)
y2 = np.convolve(x2, h)
y3 = np.convolve(x3, h)

# Nuevo eje temporal para la salida
ny = np.arange(n[0] + m[0], n[-1] + m[-1] + 1)

#%%
# GRAFICOS

## graf señales
plt.figure(figsize=(10,6))
plt.subplots_adjust(hspace= 0.5)

plt.subplot(3,1,1)
plt.stem(n, x1)
plt.title("Entrada x1[n] = u[n+1] - u[n-2]")
plt.grid(True)

plt.subplot(3,1,2)
plt.stem(n, x2)
plt.title("Entrada x2[n] = u[n+1] - u[n-2]")
plt.grid(True)

plt.subplot(3,1,3)
plt.stem(n, x3)
plt.title("Entrada x3[n] = u[n+1] - u[n-2]")
plt.grid(True)

## graf rta al impulso
plt.figure()
plt.stem(m, h)
plt.title("Respuesta impulsiva h[n] = δ[n] - δ[n-4]")
plt.grid(True)

##graf salida conv
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.stem(ny, y1)
plt.title("Salida y1[n] = x1[n] * h[n]")
plt.grid(True)

plt.subplot(3,1,2)
plt.stem(ny, y2)
plt.title("Salida y2[n] = x2[n] * h[n]")
plt.grid(True)

plt.subplot(3,1,3)
plt.stem(ny, y3)
plt.title("Salida y3[n] = x3[n] * h[n]")
plt.grid(True)

plt.tight_layout()
plt.show()
  
    
    

    
    
    
    
    
    
    
    
    
    