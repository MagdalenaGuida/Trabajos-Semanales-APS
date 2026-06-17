# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:19:30 2026

@author: magui
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


#%% LECTURA ECG 

mat_struct = sio.loadmat('./pdstestbench/ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].squeeze()   # <-- convierte (N,1) en (N,)
ecg = ecg - np.mean(ecg)
N = len(ecg)
fs = 1000

#%% FILTROS FIR

############################################
# DISEÑO FILTRO FIR VENTANAS 
############################################

nyq_frec = fs / 2
gpass = 1 #[dB]
gstop = 40 #[dB]

ws1 = .5
wp1 = .55
wp2 = 35.5
ws2 = 40


orden = 2501
b_coeffs_vent = signal.firwin(orden, [wp1, wp2], window='hamming',pass_zero=False, fs=fs)
taps = b_coeffs_vent.shape[0]


ww = np.concat([np.logspace(start = -2, stop = 0.1, num =500),
                np.linspace(start = 1.26, stop = 35, num =200),
                np.logspace(start=1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])

## ecg_filt=signal.filtfilt(b_coeffs,[1], ecg)
w, h = signal.freqz(b_coeffs_vent, [1], worN =ww, fs=fs)

############################################
# DISEÑO FILTRO FIR CUADRADOS MINIMOS
############################################

nyq_frec = fs / 2
gpass = 1 #[dB]
gstop = 40 #[dB]

ws1 = .05
wp1 = .95
wp2 = 35
ws2 = 43

bandas = np.array([0, ws1, wp1, wp2, ws2, nyq_frec])
gains= np.array([0, 0, 1, 1, 0, 0])
weight = np.array([10,2,1])
orden = 3001

b_coeffs_cmin = signal.firls(orden, 
                        bands = bandas, 
                        desired = gains, 
                        weight = weight, 
                        fs=fs)
taps = b_coeffs_cmin.shape[0]


ww = np.concat([np.logspace(start = -2, stop = 0.1, num =500),
                np.linspace(start = 1.26, stop = 35, num =200),
                np.logspace(start=1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])

w, h = signal.freqz(b_coeffs_cmin, [1], worN =ww, fs=fs)


#%% FILTROS IIR 

############################################
# DISEÑO FILTRO IIR CHEBYSHEV2
############################################

fs = 1000 
wp = [0.8, 37]
ws = [0.1, 40]
gpass = 1
gstop = 40

b_coeffs_cheby2 = signal.iirdesign(wp, ws, gpass, 45, 
                         ftype= 'cheby2',  output= 'sos', fs = fs )


ww = np.concat([np.logspace(start = -2, stop = 0.1, num =500),
                np.linspace(start = 1.26, stop = 35, num =200),
                np.logspace(start=1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])

# w,h = signal.sosfreqz(b_coeffs_cheby2, worN = ww, fs=fs) 


###########################################
##DISEÑO FILTRO IIR MAXIMA PLANICIDAD
###########################################
fs = 1000 
wp = [0.8, 37]
ws = [0.1, 40]
gpass = 1
gstop = 40

b_coeffs_butter = signal.iirdesign(wp, ws, gpass, 40, 
                         ftype= 'butter',  output= 'sos', fs = fs )

ww = np.concat([np.logspace(start = -2, stop = 0.1, num =500),
                np.linspace(start = 1.26, stop = 35, num =200),
                np.logspace(start=1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])

# w,h = signal.sosfreqz(b_coeffs_butter, worN = ww, fs=fs) 


#%%    GRAFICOS

# =============================================================================
# BLOQUE DE PLOT CON PLANTILLA DE DISEÑO (SOMBRADA)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title("Plantilla filtro PASA-BANDA")

# # 1. Dibujar la curva del filtro original (Magnitud en dB)
ax1.plot(w, 20 * np.log10(np.abs(h)), 'b', linewidth=1.8, label='Filtro diseñado')

# 2. Sombreado de la Plantilla (Zonas prohibidas / tolerancias)
# Piso y techo visual del gráfico para los rellenos

ax1.fill_between([-5,100], -125, 10, color='honeydew')
piso_grafico = -125
techo_grafico = 10

# --- Banda de parada 1 (0 a ws1) ---
ax1.fill_between([-10, 0.1], -20, techo_grafico, color='darkslategray', alpha=0.3, label='Plantilla')
ax1.plot([-10, 0.1], [-20, -20],  color='k',linestyle='--', linewidth=1, alpha=0.7) # Línea de trazo límite


# --- Banda de paso (wp1 a wp2) ---
# Zona inferior (Atenuación máxima permitida)
ax1.fill_between([1, 35], piso_grafico, -gpass, color='darkslategray', alpha=0.3)
ax1.plot([1, 35], [-gpass, -gpass],  color='k',linestyle='--', linewidth=1, alpha=0.7)

# Zona superior (Margen por encima de 0dB, ej: 3dB para evitar rizado excesivo)
ax1.fill_between([1, 35], 3, techo_grafico, color='darkslategray', alpha=0.3)
ax1.plot([1, 35], [3, 3],  color='k',linestyle='--', linewidth=1, alpha=0.7)

# --- Banda de parada 2 (ws2 a Nyquist) ---
ax1.fill_between([45, fs/2], -gstop, techo_grafico, color='darkslategray', alpha=0.3)
ax1.plot([45, fs/2], [-gstop, -gstop],  color='k',linestyle='--', linewidth=1, alpha=0.7)

##Líneas verticales que delimitan los saltos de las bandas
ax1.vlines(0.1,-20, techo_grafico, color='k', linestyle='--', alpha=0.5)
ax1.vlines( 1, piso_grafico, -1,  color='k',linestyle='--', linewidth=1, alpha=0.7)
ax1.vlines(35, piso_grafico, -1, color='k',linestyle='--', linewidth=1, alpha=0.7)
ax1.vlines( 1, 3, techo_grafico,  color='k',linestyle='--', linewidth=1, alpha=0.7)
ax1.vlines(35, 3, techo_grafico, color='k',linestyle='--', linewidth=1, alpha=0.7)
ax1.vlines(45, -gstop, techo_grafico, color='k',linestyle='--', linewidth=1, alpha=0.7)


# 3. Configuración estricta de límites y estética
ax1.set_ylabel('Amplitude in dB', color='b')
ax1.set_xlabel('Frequency [Hz]')

ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')
ax1.set_ylim([piso_grafico, techo_grafico]) 
ax1.set_xlim(-5, 100)
plt.show()



# # =============================================================================
# # DIAGRAMA DE POLOS Y CEROS
# # =============================================================================
# fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)

# # Panel Izquierdo: Plano Z
# ax_z = fig_sys.add_subplot(1, 2, 1)

# # Dibujar el círculo unitario
# circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
# ax_z.add_artist(circulo)

# # Graficar ceros (o) y polos (x)
# ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=8, fillstyle='none', label='Ceros')
# ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, mew=2, label='Polos')

# ax_z.axhline(0, color='black', linewidth=0.5)
# ax_z.axvline(0, color='black', linewidth=0.5)
# ax_z.set_title('Diagrama de Polos y Ceros (Plano z)')
# ax_z.set_xlabel('Parte Real')
# ax_z.set_ylabel('Parte Imaginaria')
# ax_z.axis('equal')
# ax_z.set_xlim([-1.2, 1.2])
# ax_z.set_ylim([-1.2, 1.2])
# ax_z.grid(True, alpha=0.5)
# ax_z.legend()


# # =============================================================================
# # DIAGRAMA DE RETARDO DE GRUPO
# # =============================================================================

# # Panel Derecho: Retardo de Grupo
# ax_gd = fig_sys.add_subplot(1, 2, 2)

# # Calcular el retardo de grupo usando scipy
# # sos2tf convierte la matriz sos a polinomios b, a solo para el cálculo rápido del retardo
# #w_gd, gd = sig.group_delay(sig.sos2tf(sos), w=1024, fs=fs)

# ax_gd.plot(gd, gd, 'm', linewidth=2)
# ax_gd.set_title('Retardo de Grupo (Group Delay)')
# ax_gd.set_xlabel('Frecuencia [Hz]')
# ax_gd.set_ylabel('Retardo [Muestras]')
# #ax_gd.set_xlim(0, fs/2)
# ax_gd.grid(True, alpha=0.5)

# plt.show()


#%% APLICACION DE LOS FILTROS

# FIR 

ecg_fir_vent = signal.filtfilt(b_coeffs_vent, [1], ecg)
ecg_fir_cmin = signal.filtfilt(b_coeffs_cmin, [1], ecg)
ecg_iir_cheby2 = signal.sosfiltfilt(b_coeffs_cheby2, ecg)
ecg_iir_butter = signal.sosfiltfilt(b_coeffs_butter, ecg)


# Mostrar solo 5 segundos
t = np.arange(N) / fs
t_ini = 0
t_fin = 5
mask = (t >= t_ini) & (t <= t_fin)

fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True, tight_layout=True)

# ── ECG original ──────────────────────────────────────
axes[0].plot(t[mask], ecg[mask],
             color='gray', linewidth=0.6)
axes[0].set_title('ECG Original')

# ── FIR Ventanas ──────────────────────────────────────
axes[1].plot(t[mask], ecg_fir_vent[mask],
             color='blue', linewidth=0.8)
axes[1].set_title('FIR Ventanas (Hamming) — orden 2501')

# ── FIR Cuadrados Mínimos ─────────────────────────────
axes[2].plot(t[mask], ecg_fir_cmin[mask],
             color='darkorange', linewidth=0.8)
axes[2].set_title('FIR Cuadrados Mínimos (firls) — orden 1301')

# ── IIR Butterworth ───────────────────────────────────
axes[3].plot(t[mask], ecg_iir_butter[mask],
             color='red', linewidth=0.8)
axes[3].set_title('IIR Butterworth')

# ── IIR Chebyshev 2 ───────────────────────────────────
axes[4].plot(t[mask], ecg_iir_cheby2[mask],
             color='green', linewidth=0.8)
axes[4].set_title('IIR Chebyshev 2')
axes[4].set_xlabel('Tiempo (s)')

for ax in axes:
    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.4)

plt.suptitle('Comparación de filtros aplicados al ECG',
             fontsize=13, fontweight='bold')
plt.show()


cant_muestras = N   # len(ecg)
ecg_one_lead  = ecg

# Demora (0 si usás filtfilt)
demora = 0

###################################
# Regiones CON ruido
###################################
regs_interes = (
    [4000,  5500],
    [10000, 11000],
)

for ii in regs_interes:
    zoom_region = np.arange(np.max([0, ii[0]]),
                            np.min([cant_muestras, ii[1]]),
                            dtype='uint')

    plt.figure(figsize=(12, 4))
    plt.plot(zoom_region, ecg_one_lead[zoom_region],
             label='ECG original', linewidth=2, color='gray')
    plt.plot(zoom_region, ecg_fir_vent[zoom_region + demora],
             label='FIR Ventanas', linewidth=1.2, color='blue')
    plt.plot(zoom_region, ecg_iir_butter[zoom_region + demora],
             label='Butterworth', linewidth=1.2, color='red')
    plt.plot(zoom_region, ecg_iir_cheby2[zoom_region + demora],
             label='Chebyshev 2', linewidth=1.2, color='green')

    plt.title(f'ECG con ruido — muestras {ii[0]} a {ii[1]}')
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.gca().set_yticks(())  
    plt.tight_layout()
    plt.show()

















