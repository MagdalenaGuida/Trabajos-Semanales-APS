# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:47:20 2026

@author: magui
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio


## FUNCION BW
def BW(f_welch, pxx_welch, cota_inf=0.005, cota_sup=0.99):
    
    pot_tot = np.sum(pxx_welch)
    potencia_normalizada = np.cumsum(pxx_welch) / pot_tot
    
    fmin = np.where(potencia_normalizada >= cota_inf)[0][0]
    fmax = np.where(potencia_normalizada >= cota_sup)[0][0]
    frec_min = f_welch[fmin]
    frec_max = f_welch[fmax]
    bw = frec_max - frec_min
    
    return frec_min, frec_max, bw

#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz
ecg = np.load('./pdstestbench/ecg_sin_ruido.npy')
N_ecg =  len(ecg)
df_ecg = fs_ecg / N_ecg

plt.figure()
plt.plot(ecg)
plt.title('ECG')
plt.xlabel('Tiempo [s]')
plt.grid(True)
plt.show()

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz
ppg = np.load('./pdstestbench/ppg_sin_ruido.npy')
N_ppg = len(ppg)
df_ppg = fs_ppg / N_ppg

plt.figure()
plt.plot(ppg)
plt.title('PPG')
plt.xlabel('Tiempo [s]')
plt.grid(True)
plt.show()



#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('./pdstestbench/la cucaracha.wav')
N_audio = len(wav_data)
df_audio = fs_audio / N_audio

plt.figure()
plt.plot(wav_data)
plt.title('AUDIO')
plt.xlabel('Tiempo [s]')
plt.grid(True)
plt.show()


#%%
#------------------------ DEF WELCH---------------------------------------------#

## WELCH ECG
cant_promedio1 = 20
nperseg1 = ecg.shape[0]// cant_promedio1
win1 = 'hamming'
mult_nfft = 4
f_welch1, pxx_welch1 = sig.welch(ecg, fs_ecg, nperseg= nperseg1, window= win1, axis = 0, nfft = mult_nfft * nperseg1)


## WELCH PPG
ppg_fft = np.fft.fft(ppg, axis = 0) /N_ppg
nperseg2 = ppg.shape[0] // cant_promedio1
win2 = 'hann'
f_welch2, pxx_welch2 = sig.welch(ppg, fs_ppg, nperseg= nperseg2, axis=0, window= win1,  nfft = mult_nfft * nperseg2)

## WELCH AUDIO
cant_promedio3 = 30
nperseg3 = wav_data.shape[0] // cant_promedio3
win3 = 'bartlett'
f_welch3, pxx_welch3 = sig.welch(wav_data, fs_audio, nperseg= nperseg3, axis=0, window= win3, nfft =mult_nfft * nperseg3)



#%% PUNTO DOS: 
    
frec_min1, frec_max1, bw_ecg = BW(f_welch1, pxx_welch1)
frec_min2, frec_max2, bw_ppg = BW(f_welch2, pxx_welch2)
frec_min3, frec_max3, bw_audio = BW(f_welch3, pxx_welch3)

#%% GRAFICOS 

## TABLA
##Crear tabla con los encabezados fijos
tabla = [
    ["SEÑAL",   "fs [Hz]",   "frec -> 0.5% [Hz]",  "frec -> 99% [Hz]",  "BW [Hz]"],
    ["ECG", fs_ecg, f"{frec_min1:.2f}", f"{frec_max1:.2f}", f"{bw_ecg:.2f}"],
    ["PPG", fs_ppg, f"{frec_min2:.2f}", f"{frec_max2:.2f}", f"{bw_ppg:.2f}"],
    ["AUDIO", fs_audio, f"{frec_min3:.2f}", f"{frec_max3:.2f}", f"{bw_audio:.2f}"],
]

# Mostrar como imagen
fig, ax = plt.subplots(figsize=(12, 2 + len(tabla)*0.5))
ax.axis('tight')
ax.axis('off')
plt.title("Estimacion ANCHO DE BANDA", fontsize=11, pad=1)
table = ax.table(cellText=tabla, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)
 
plt.tight_layout()
plt.show()

# GRAFICOS 
plt.figure()
plt.plot(f_welch1, pxx_welch1)
plt.title('ECG:  Metodo WELCH')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.xlim(0,50)
plt.axvline(x=frec_min1, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min1:.2f} Hz')
plt.axvline(x=frec_max1, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max1:.2f} Hz')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(f_welch2, pxx_welch2)
plt.title('PPG: Metodo WELCH ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.xlim(0,15)
plt.axvline(x=frec_min2, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min2:.2f} Hz')
plt.axvline(x=frec_max2, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max2:.2f} Hz')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(f_welch3, pxx_welch3)
plt.title('AUDIO, La Cucaracha: Metodo WELCH ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.xlim(0, 2500)
plt.axvline(x=frec_min3, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min3:.2f} Hz')
plt.axvline(x=frec_max3, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max3:.2f} Hz')
plt.legend()
plt.grid(True)
plt.show() 

# ## GRAFICOS ZOOM
# plt.figure()
# plt.plot(f_welch1[bfrec1], 10* np.log10(pxx_welch1[bfrec1]))
# plt.title('ECG: Espectro de Potencia- Metodo WELCH (zoom)')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Potencia [dB]')
# plt.xlim(-2, 35)
# plt.ylim(37, 60)
# plt.axvline(x=frec_min1, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min1:.2f} Hz')
# plt.axvline(x=frec_max1, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max1:.2f} Hz')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.plot(f_welch2[bfrec2], 10* np.log10(pxx_welch2[bfrec2]))
# plt.title('PPG: Espectro de Potencia- Metodo WELCH(zoom) ')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Potencia [dB]')
# plt.xlim(-1, 7)
# plt.ylim(10, 60)
# plt.axvline(x=frec_min2, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min2:.2f} Hz')
# plt.axvline(x=frec_max2, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max2:.2f} Hz')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.plot(f_welch3[bfrec3], 10* np.log10(pxx_welch3[bfrec3]))
# plt.title('AUDIO, silbido: Espectro de Potencia- Metodo WELCH (zoom) ')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Potencia [dB]')
# plt.xlim(2500, 7500)
# plt.ylim(-100, -50)
# plt.axvline(x=frec_min3, color='red', linestyle='--', linewidth=1, label=f'f_min = {frec_min3:.2f} Hz')
# plt.axvline(x=frec_max3, color='red', linestyle='--', linewidth=1, label=f'f_max = {frec_max3:.2f} Hz')
# plt.legend()
# plt.grid(True)
# plt.show() 