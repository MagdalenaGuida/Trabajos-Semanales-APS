# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:02:28 2026

@author: magui
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt


#----------------------------------- Datos de la simulación---------------------------

N = 1000 # cantidad de muestras
fs =  N # frecuencia de muestreo (Hz)
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral (=1)
dc = 0
ph = 0

# Datos del ADC
Vf = 2      #Vf max = 2  y Vf min= -2, rango TOTAL=4
B4= 4       #B-bits para a)
B8 = 8
B16 = 16
B = B4

# Definicion funciónes seno
def func_sen(Vmax=1, dc = dc, ff=1, ph=ph, nn = N, fs=fs):
    N= np.arange(nn)
    tt = N/fs    
    xx = Vmax * np.sin(2 * np.pi * ff * tt + ph).reshape(nn,1) + dc
    return tt, xx

## Definicion funcion: Verificacion Parseval
def verf_Pars(xx):
    N = len(xx)
    Pt = np.sum(np.abs(xx)**2) 
    Pf = 1/N * np.sum(np.abs(np.fft.fft(xx))**2)  
    verf_Pars = Pt - Pf
    return verf_Pars

## Definicion funcion: Verificacion Piso de ruido
def verf_piso_de_ruido(xx):
    N = len(xx)
    Pt = np.var(xx) 
    Pf =np.sum(np.abs( 1/N * np.fft.fft(xx))**2) 
    Pt_dB = 10*np.log10(Pt)
    Pf_dB = 10*np.log10(Pf)
    verf_piso_de_ruido = Pt_dB - Pf_dB
    return verf_piso_de_ruido

#%%

## DEFINICION SEÑAL
tt, xx = func_sen(ff = 1, Vmax= np.sqrt(2)) # amp, dc, frec,p, N, fs
xx = xx.flatten()
# Normalizacion
xx=xx/np.std(xx)  
# Calculo de potencia DE LA SEÑAL, se espera Ps =1
Ps = np.mean(xx**2)

# POTENCIA
q = (2*Vf)/(2**B)# paso de cuantización de q Volts = (Vmax - Vmin)/2^B
Pq = q**2/12 # Watts
k01 = 1/10 
k1 = 1
k10 = 1
k = k10       # escala de la potencia de ruido analógico
Pn = Pq * k   # potencia del ruido analogico

## RUIDO
nn=np.random.normal(0, np.sqrt(Pn), N) #señal de ruido analogico
# media: 0, desvio: raiz de ruido analogico: ruido cuant * (K=1) 
incorr = sig.convolve(nn,nn)
sr = xx + nn # señal analógica de entrada al ADC (con ruido analógico)

## RUIDO CUANTIZADO
srq = np.round(sr/q)*q  #ADC OUT
#Ej: si q=0.1 y sr=0,43, hago 0.43/0.1=4.3, redondeo para -> 4, entonces hago4*(q=0.1)=0.4
nq =  srq-sr #  ruido de cuantización || aca se hace la diferencia entre la señal y la señal cuantizada (ambas con ruido)

# CALCULO FFT + VECT FRECS
#Uso la fft aplicada sobre señales en el tiempo para verlas en el espectro 
ft_SR =  1/N*np.fft.fft(sr)          #sr: analogica con ruido 
ft_Srq = 1/N*np.fft.fft(srq)         #srq: cuantizada
ft_Xx =  1/N*np.fft.fft(xx)          #Xx: senoidal limpia
ft_Nq =  1/N*np.fft.fft(nq)          #Nq: ruido cuantizado= sr-srq
ft_Nn =  1/N*np.fft.fft(nn)          #Nn: ruido analogico 

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)   #vector de frecs desde 0 hasta fs-1 (en este caso pq N=fs) 

bfrec = ff <= fs/2    #bfrec: filtro booleano que te deja solo las frecs hasta fs/2 = frc de nyquist
nn_prom = np.mean(np.abs(ft_Nq)**2)  
nq_prom = np.mean(np.abs(ft_Nn)**2)

#%%
# # GRAFICOS 
# #GRAF 1: SEÑAL + RUIDO
# plt.figure()
# plt.plot(tt, srq, color='blue', label='Señal cuantizada (ADC out)')
# plt.plot(tt, sr, color='g',alpha= 0.7, ls='dotted',marker='o',markerfacecolor='none',markeredgecolor='g',markersize=2, label=' Sr= xx + nn  (ADC in)')
# plt.plot(tt, xx, color='orange', ls='--', label='señal limpia')
# plt.title('Señal muestreada por un ADC de {:d} bits - q = {:3.3f} V'.format(B, q) )
# plt.xlabel('tiempo [segundos]')
# plt.ylabel('Amplitud [V]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.grid(True)
# plt.show()
  
## GRAF 2: ESPERCTRO 
piso_ruido_analogico = 10* np.log10(2* nn_prom)
piso_ruido_digital = 10* np.log10(2* nq_prom)
plt.figure()
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2),color='blue', label='señal cuantizada (ADC out)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Xx[bfrec])**2), color='orange', ls='--', label='Senoidal limpia' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label=' señal analogica  (ADC in)' )
plt.hlines(y= piso_ruido_analogico, xmin=0, xmax=500, colors='maroon', linestyles='--',  label= f'Piso de ruido analogico: {piso_ruido_analogico}dB')
plt.hlines(y= piso_ruido_digital , xmin=0, xmax=500, colors='pink', linestyles='--',  label= f'Piso de ruido digital: {piso_ruido_digital}dB')
plt.title('Señal muestreada por un ADC de {:d} bits - q = {:3.10f} V'.format(B, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
#plt.ylim(-160, 0)
plt.show()

# ## GRAF 3: HISTOGRAMA 
# bins =10
# xlim = q/2
# ylim = N / bins
# plt.figure()
# plt.title('Ruido de cuantizacion para {:d} bits - q = {:3.3f} V'.format(B, q) )
# plt.hist(nq, bins=bins, color='blue', alpha=0.5)
# plt.hlines(y= ylim, xmin=-xlim, xmax=xlim, colors='r', linestyles='--')
# plt.vlines(x=-xlim, ymin = 0, ymax = ylim, colors='r', linestyles='--')
# plt.vlines(x=xlim, ymin = 0, ymax = ylim, colors='r', linestyles='--')
# plt.grid(True)
# plt.show()


# #%%
# # Valores de K y B a comparar
# Ks = [1/10, 1, 10]
# Bs = [4, 8, 16]

# # Encabezado de la tabla
# print("{:<8} {:<8} {:<20} {:<20}".format("Bits", "K", "Piso analógico (dB)", "Piso digital (dB)"))
# print("-"*60)

# for B in Bs:
#     q = (2*Vf)/(2**B)
#     pot_ruido_cuant = q**2/12
    
#     for K in Ks:
#         # Potencia de ruido analógico
#         pot_ruido_analog = pot_ruido_cuant * K
#         sigma = np.sqrt(pot_ruido_analog)
        
#         # Señal senoidal normalizada
#         tt, xx = func_sen(Vmax=np.sqrt(2))
#         xx = xx/np.std(xx)
        
#         # Ruido analógico
#         nn = np.random.normal(0, sigma, N)
#         sr = xx + nn
        
#         # Cuantización
#         srq = np.round(sr/q)*q
#         nq = srq - sr
        
#         # FFTs
#         ft_Nn = 1/N*np.fft.fft(nn)
#         ft_Nq = 1/N*np.fft.fft(nq)
        
#         # Pisos de ruido (promedio espectral)
#         piso_analogico = 10*np.log10(2*np.mean(np.abs(ft_Nn)**2))
#         piso_digital   = 10*np.log10(2*np.mean(np.abs(ft_Nq)**2))
        
#         # Mostrar fila
#         print("{:<8} {:<8} {:<20.2f} {:<20.2f}".format(B, K, piso_analogico, piso_digital))

# #%%
# ## BONUS

# # EFECTO ALIASING 

# falias1 = 400
# tt1, xx1 = func_sen(ff=falias1, Vmax=np.sqrt(2))
# xx1 = xx1.flatten()
# xx1 = xx1/np.std(xx1)
# E1 = np.mean(xx1**2)

# falias2 = 600
# tt2, xx2 = func_sen(ff= falias2, Vmax= np.sqrt(2))
# xx2 = xx2.flatten()
# xx2 =xx2/np.std(xx2)  
# E2 = np.mean(xx2**2)

# #Uso la fft aplicada sobre señales en el tiempo para verlas en el espectro 
# ft_xx1 = 1/N*np.fft.fft(xx1)           
# ft_xx2 = 1/N*np.fft.fft(xx2)         


# # grilla de sampleo frecuencial
# ff = np.linspace(0, (N-1)*df, N)   #vector de frecs desde 0 hasta fs-1 (en este caso pq N=fs) 

# bfrec = ff <= fs/2    #bfrec: filtro booleano que te deja solo las frecs hasta fs/2 = frc de nyquist

# ### GRAFICO
# plt.figure()
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx1[bfrec])**2), color='orange', label=f'Senoidal f= {falias1}Hz' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx2[bfrec])**2), color= 'green', linestyle='--', label=f'Senoidal f= {falias2}Hz')
# plt.title('EFECTO ALIASING')
# plt.ylabel('Densidad de Potencia [dB]')
# plt.xlabel('Frecuencia [Hz]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.grid(True)
# plt.show()

# #%%  BONUS: SNR <-> B-BITS 

# SNR = [ 
#     ['', 1/10, 1, 10],
#     [4, 35.84, 25.84, 15.84],
#     [8, 59.93, 49.93, 39.93],
#     [16, 108.09, 98.09, 88.09]
# ]

# fig, ax = plt.subplots()
# ax.axis('tight')
# ax.axis('off')
# tabla = ax.table(cellText=SNR, loc='center', cellLoc='center')
# tabla.auto_set_font_size(False)
# tabla.set_fontsize(14)
# tabla.scale(1.2, 3)
# plt.title("SNR [dB] para cambios de k VS B-bits", fontsize=14)
# plt.show()
