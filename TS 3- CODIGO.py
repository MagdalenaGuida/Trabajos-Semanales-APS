# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:59:56 2026

@author: magui
"""

import numpy as np 
from numpy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt


fs = 1000  #Frecuencia de muestreo
N = 1000   #Número de muestras
ts = 1 / fs  #Tiempo entre muestras correlativas
# tiempo total de muestreo = N*ts = N * 1/fs = 1s
df = fs/N              #resolucion espectral
df2 = fs / (9*N)
dc = 0       #Desplazamiento vertical [V]  
ph = 0       #FASE = Desplazamiento horizontal [rad]
ff = 0

# Definicion funciónes seno
def func_sen(Vmax=1, nn=N, fs=fs, ff=ff, dc=dc, ph=ph):
    N= np.arange(nn)
    tt = N/fs
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

##Parseval
def id_Parseval(xx):
    N = len(xx)
    Et = np.sum(np.abs(xx)**2) 
    X_fft= np.abs(np.fft.fft(xx))
    Ef = 1/N * np.sum(X_fft**2) 
    verf_Pars = Et - Ef
    return verf_Pars



#%% DEFS FUNCIONES

tt, xx = func_sen(ff = N/4 * (fs/N), Vmax = np.sqrt(2))
tt2, xx2 = func_sen(ff= (N/4 + 1/4) *  (fs/N), Vmax = np.sqrt(2) ) 
tt3, xx3 = func_sen(ff = (N/4 + 1/2) * (fs/N), Vmax = np.sqrt(2))

## Verificacion potencia unitaria 
potencia = np.mean(np.abs(xx**2))
potencia2 = np.mean(np.abs(xx2**2))
potencia3 = np.mean(np.abs(xx3**2))

X1_fft = np.abs(1/N * np.fft.fft(xx))# / (np.sqrt(2) /2)
X2_fft = np.abs(1/N * np.fft.fft(xx2))
X3_fft = np.abs(1/N * np.fft.fft(xx3))
frecs = np.arange(N) * fs / N

## SEÑALES SENO + ZERO-PADDING 
zeros = np.zeros(9*N, dtype= complex)
xz = np.concat((xx, zeros))
xz2 = np.concat((xx2,zeros))
xz3 = np.concat((xx3, zeros))

##fft + vector frecs
Xz_fft = np.abs(1/(10*N) * np.fft.fft(xz))
Xz2_fft = np.abs(1/(10*N) * np.fft.fft(xz2))
Xz3_fft = np.abs(1/(10*N) * np.fft.fft(xz3))
frecs2 = np.arange(10*N) * fs / (10*N)

#%% GRAFICO: MODULO

plt.figure()
plt.plot(frecs[:N//2], 10* np.log(X1_fft[:N//2]**2) ,':', color = 'b', label='ff1= df')
plt.plot(frecs[:N//2], 10* np.log(X2_fft[:N//2]**2), ':', color = 'forestgreen', label = 'ff2= df + 1/4')
plt.plot(frecs[:N//2], 10* np.log(X3_fft[:N//2]**2), ':', color = 'red', label = 'ff3= df + 1/2')
plt.legend(loc = 'lower left')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.title('Espectro de la señal')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(frecs[:N//2], 10* np.log(X1_fft[:N//2]**2) ,':', color = 'b', label='ff1= df')
plt.plot(frecs[:N//2], 10* np.log(X2_fft[:N//2]**2), ':', color = 'forestgreen', label = 'ff2= df + 1/4')
plt.plot(frecs[:N//2], 10* np.log(X3_fft[:N//2]**2), ':', color = 'red', label = 'ff3= df + 1/2')
plt.legend(loc = 'lower left')
plt.xlim(248,252)
plt.ylim(-22,-6)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.title('Espectro de la señal [ZOOM]')
plt.grid(True)

Ef1 = 1/N * np.sum(X1_fft**2) 
Ef2 = 1/N * np.sum(X2_fft**2) 
Ef3 = 1/N * np.sum(X3_fft**2) 

#%% GrAFICOS ZERO-PADDING 

plt.figure()
plt.plot(frecs2[:9*N//2], 10* np.log(Xz_fft[:9*N//2]**2) ,':', color = 'lightblue', label='ff1 + ZP')
plt.plot(frecs2[:9*N//2], 10* np.log(Xz2_fft[:9*N//2]**2) ,':', color = 'lime', label='ff2 + ZP')
plt.plot(frecs2[:9*N//2], 10* np.log(Xz3_fft[:9*N//2]**2) ,':', color = 'pink', label='ff3 + ZP')
plt.legend(loc = 'lower left')
plt.xlim(248,252.5)
plt.ylim(-95, -45)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('dB')
plt.title('Fig. 5: Espectro de la señal + zeropadding [ZOOM]')
plt.grid(True)

senales = [
    ("ff1 + zero-padding", frecs2[:9*N//2], 10* np.log(Xz_fft[:9*N // 2]**2), 'lightblue'),
    ("ff1 ", frecs[:N//2], 10* np.log(X1_fft[:N//2]**2), 'b'),
    ("ff2 + zero-padding", frecs2[:9*N//2], 10* np.log(Xz2_fft[:9*N // 2]**2), 'lime'),
    ("ff2", frecs[:N//2], 10* np.log(X2_fft[:N//2]**2), 'forestgreen'),
    ("ff3 + zero-padding", frecs2[:9*N//2], 10* np.log(Xz3_fft[:9*N // 2]**2), 'pink'),
    ("ff3", frecs[:N//2], 10* np.log(X3_fft[:N//2]**2), 'r'),
]

# Crear 3 subplots (uno por grupo)
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# Graficar señales agrupadas
grupos = [(0, 1), (2, 3), (4, 5)]

for ax, (i1, i2) in zip(axs, grupos):
    for i in [i1, i2]:
        titulo, tiempo, señal, color = senales[i]
        ax.plot(tiempo, señal, color=color, label=titulo)
    ax.set(xlabel="Frecuencia [Hz]", ylabel="dB")
    ax.grid(True)
    ax.legend()
fig.suptitle("Comparacion señales vs señales + zero-padding", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])


#%% ESTADISTICA

# PARSEVAL 
V1 = id_Parseval(xx)
V2 = id_Parseval(xx2)
V3 = id_Parseval(xx3)
V4z = id_Parseval(xz)
V5z = id_Parseval(xz2)
V6z = id_Parseval(xz3)

print(f"Id de Parseval para señal 1= {V1}")
print(f"Id de Parseval para señal 2= {V2}")
print(f"Id de Parseval para señal 3= {V3}")
print(f"Id de Parseval para señal 4= {V4z}")
print(f"Id de Parseval para señal 5= {V5z}")
print(f"Id de Parseval para señal 6= {V6z}")


#%% BONUSSSS

# Definimos el rango de n para la entrada
n = np.arange(-10, 40)

# Función escalón
u = lambda x: np.heaviside(x, 1)

W0 = 1
Ts = 1

# Entrada: x[n] = u[n+1] - u[n-2]
x1 = u(n+1) - u(n-2)
x2 = (1/2)**n * u(n)
x3 = np.cos(W0 * n * Ts)

# # Definimos el rango de n para h[n]
m = np.arange(0, 50)

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
NY = len(ny)

Y1_fft = np.abs(np.fft.fft(y1)) 
Y2_fft = np.abs(np.fft.fft(y2))
Y3_fft = np.abs(np.fft.fft(y3))
frecsLTI = np.arange(NY) * (2* np.pi) / NY


#%% GRAFS

plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(frecsLTI[:NY//2], 20*np.log10(Y1_fft[:NY//2]))
plt.title("FFT de y1[n]")
plt.xlabel("W [rad/muestra]")
plt.ylabel("[dB]")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(frecsLTI[:NY//2], 20*np.log10(Y2_fft[:NY//2]))
plt.title("FFT de y2[n]")
plt.xlabel("W [rad/muestra]")
plt.ylabel("[dB]")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(frecsLTI[:NY//2], 20*np.log10(Y3_fft[:NY//2]))
plt.title("FFT de y3[n]")
plt.xlabel("W [rad/muestra]")
plt.ylabel("[dB]")
plt.grid(True)

plt.tight_layout()
plt.show()