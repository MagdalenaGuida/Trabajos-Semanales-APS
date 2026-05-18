# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:17:37 2026

@author: magui
"""
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig 


#%%DEFINO PARAMETROS

SNR = 10 #[dB]
R = 200  #cant de realizaciones 
N = 1000 # cant de muestras ----------> esto me da dim NxR 
fs= N
ts = 1/fs
a0 = np.sqrt(2) 
w0 = N / 4
df= fs/N

vect_t = np.arange(0, 1, 1/N).reshape(N, 1)
tt = np.tile(vect_t, (1, R))
fr = np.random.uniform(-2,2,size = (1,R))

w1 = w0 + fr* df

#SEÑAL LIMPIA 
S = a0*np.sin(w1 * tt * 2 * np.pi) # mult matricia entre w1 y tt = S[1000,200]=[N,R]

#DEF RUIDO 

sigma_n_cuadrado = ((a0**2)/2)/(10**(SNR/10))         ## OBS!  ESTA AL CUADRADO
sigma_n = np.sqrt(sigma_n_cuadrado)
SNR_verf = 10*np.log10((a0**2 / 2) / sigma_n_cuadrado)

na=np.random.normal(0, sigma_n , size=(N,R))  #señal de ruido analogico
# media: 0, desvio: raiz de ruido analogico


#%% DEF SEÑAL + VENTANAS 

# VENT RECTANGULAR 
X = S + na    
# VENT FLATTOP
vent_ft = sig.windows.flattop(N).reshape(-1 ,1)
XFT = X * vent_ft
# VENT BLACKMANHARRIS 
vent_bmh = sig.windows.blackmanharris(N).reshape(-1,1)
XBMH = X * vent_bmh
# VENT HAMMING
vent_hm = sig.windows.hamming(N).reshape(-1,1)
XH = X * vent_hm


#%% DEF FFT 

X_FFT0 = 1 / N * np.fft.fft(X, n= N, axis=0)
X_FFT = np.abs( X_FFT0[:N//2, :])
XFT_FFT0 = 1 / N * np.fft.fft(XFT,n= N, axis = 0)
XFT_FFT = np.abs(XFT_FFT0[ :N//2, :])
XBMH_FFT0 = 1 / N * np.fft.fft(XBMH,n= N, axis = 0)
XBMH_FFT = np.abs(XBMH_FFT0[ :N//2, :])
XH_FFT0 = 1 / N * np.fft.fft(XH,n= N, axis = 0)
XH_FFT = np.abs(XH_FFT0[ :N//2, :])


#%% DEFINIR ESTIMADOR DE AMPLITUD = a1

# valor verdadero = a0
a_R = np.abs(X_FFT0[N//4, :])
a_FT = np.abs(XFT_FFT[N//4, :])
a_BMH =  np.abs(XBMH_FFT0[N//4, :])
a_H = np.abs(XH_FFT0[N//4, :])

# SESGO 
# Calculo el valor promedio de cada estimador --> 'valor esperado'
E_a_R = np.mean(a_R)
E_a_FT = np.mean(a_FT)
E_a_BMH = np.mean(a_BMH)
E_a_H = np.mean(a_H)

#Sesgo = valor promedio - valor real 
S_a_R = E_a_R - a0
S_a_FT = E_a_FT - a0
S_a_BMH = E_a_BMH - a0
S_a_H = E_a_H - a0

#VARIANZA 
V_a_R = np.var(a_R)
V_a_FT = np.var(a_FT)
V_a_BMH = np.var(a_BMH)
V_a_H = np.var(a_H)

#%% Definir estimador de Frecs = omega

w_R = np.argmax(X_FFT, axis=0) * df  # argmax, axis=0 te tira la posicion de maximo por columna, y dsp se lo mult por la df para N/2
w_FT = np.argmax(XFT_FFT, axis =0)* df
w_BMH = np.argmax(XBMH_FFT, axis =0)* df 
w_H = np.argmax(XH_FFT, axis=0)* df 

#SESGO 
#valor real = w0
E_w_R = np.mean(w_R)   #mean = valor promedio 
E_w_FT = np.mean(w_FT)
E_w_BMH = np.mean(w_BMH)
E_w_H = np.mean(w_H)

#Sesgo = valor promedio - valor real 
S_w_R = E_w_R - w0
S_w_FT = E_w_FT - w0
S_w_BMH = E_w_BMH - w0
S_w_H = E_w_H - w0

#VARIANZA
V_w_R = np.var(w_R)
V_w_FT = np.var(w_FT)
V_w_BMH = np.var(w_BMH)
V_w_H = np.var(w_H)

#%% GRAFICOS
##Crear tabla con los encabezados fijos
tabla = [
    ["Ventana ",     "SESGO a",      "VAR a",      "SESGO omega",      "VAR omega"],
    ["Rectangular", S_a_R, V_a_R, S_w_R , V_w_R],
    ["Flattop", S_a_FT,  V_a_FT, S_w_FT, V_w_FT],
    ["Blackmanharris", S_a_BMH,  V_a_BMH, S_w_BMH, V_w_BMH],
    ["Hamming", S_a_H,  V_a_H, S_w_H, V_w_H]
]

# Mostrar como imagen
fig, ax = plt.subplots(figsize=(12, 2 + len(tabla)*0.5))
ax.axis('tight')
ax.axis('off')
plt.title(f"Sesgo y Varianza para Estimadores: a y omega - {SNR} SNR", fontsize=11, pad=1)
table = ax.table(cellText=tabla, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

plt.tight_layout()
plt.show()

#%% GRAFICOS 


####Histograma omega
plt.figure()
plt.hist(w_R, bins=10, color='red',fill= 'True', alpha=0.5, label="Estimador sin ventanear")
plt.hist(w_FT, bins=10,color='green',fill= 'True', alpha=0.5, label="Estimador ventana Flattop")
plt.hist(w_BMH, bins=10, color='blue',fill= 'True', alpha=0.5,label="Estimador ventana BlackmanHarris")
plt.hist(w_H, bins=10,  color='pink',fill= 'True', alpha=0.5, label="Estimador ventana Hamming")
plt.title(f"Histograma de frecuencias estimadas - SNR: {SNR} dB")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.grid(True)
plt.legend()
plt.show()



# Histograma a
plt.figure()
plt.hist(a_R, bins=10, color='red', alpha=0.5, label="Estimador sin ventanear") #Bins: resolucion espectral del histograma; conteo relativo. ANCHURA de los valores.
plt.hist(a_FT, bins=10, color='green', alpha=0.5, label="Estimador ventana Flattop")
plt.hist(a_BMH, bins=10, color='blue', alpha=0.5, label="Estimador ventana BlackmanHarris")
plt.hist(a_H, bins=10, color='pink', alpha=0.5, label="Estimador ventana Hamming")
plt.title(f"Histograma de amplitudes estimadas - SNR: {SNR} dB")
plt.xlabel("Amp")
plt.ylabel("Cantidad de ocurrencias")
plt.grid(True)
plt.legend()
plt.show()






