# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:27:55 2026

@author: magui
"""

"""
Análisis de filtro digital T(z) = (z^3 + z^2 + z + 1) / z^3)
  - Polos y ceros
  - Respuesta en frecuencia T(e^jw)
  - Módulo en dB vs frecuencia [rad/s]
  - Fase en [rad] vs frecuencia [rad/s]
  - Diagrama de polos y ceros

Uso:
    python analisis_Tz.py

    Opcionalmente podés cambiar 'num' y 'den' para analizar otro filtro.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# ─────────────────────────────────────────────
# PARÁMETROS — solo modificar esta sección
# ─────────────────────────────────────────────

num = [ 1, 1, 1, 1]   # coeficientes del numerador  (potencias de z, mayor a menor)
den = [1, 0, 0, 0 ]   # coeficientes del denominador

titulo = r"$T(z) = \dfrac{z^3 + z^2 + z + 1}{z^3}$"   # título del gráfico
label_polo = "p=0 (x3) "                                # multiplicidad del polo en origen

# ─────────────────────────────────────────────
# CÁLCULOS
# ─────────────────────────────────────────────

fs = 2 * np.pi
N  = 2048

zeros = np.roots(num)
poles = np.roots(den)

w, H      = signal.freqz(num, den, worN=N, fs=fs)
mag_dB    = 20 * np.log10(np.abs(H) + 1e-12)
phase_rad = np.unwrap(np.angle(H))

# Retardo de grupo: (grado_num - grado_den) / 2  →  pendiente teórica de fase
retardo = (len(num) - 1) / 2   # = (M-1)/2
phase_teo = -retardo * w

# ─────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────

DARK   = "#0f1117"
PANEL  = "#1a1d27"
GRID   = "#2e3147"
TEXT   = "#d0d4e8"
BLUE   = "#4a9eff"
RED    = "#ff5f5f"
GREEN  = "#3ecf8e"
YELLOW = "#f9c74f"

fig = plt.figure(figsize=(13, 10), facecolor=DARK)
fig.suptitle(titulo, fontsize=16, color="white", y=0.97)

gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38,
                       left=0.09, right=0.96, top=0.90, bottom=0.08)

ax_pz  = fig.add_subplot(gs[0, 0])
ax_mag = fig.add_subplot(gs[0, 1])
ax_ph  = fig.add_subplot(gs[1, :])

xticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["0", "π/4", "π/2", "3π/4", "π"]

for ax in [ax_pz, ax_mag, ax_ph]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)

# ── Polos y ceros ─────────────────────────────
ax_pz.set_title("Diagrama de polos y ceros", fontsize=11)
ax_pz.set_xlabel("Re(z)")
ax_pz.set_ylabel("Im(z)")
ax_pz.set_aspect("equal")
ax_pz.axhline(0, color=GRID, linewidth=0.8)
ax_pz.axvline(0, color=GRID, linewidth=0.8)

theta = np.linspace(0, 2 * np.pi, 400)
ax_pz.plot(np.cos(theta), np.sin(theta),
           color="#6070a0", linewidth=0.9, linestyle="--", label="|z|=1")

ax_pz.scatter(zeros.real, zeros.imag,
              s=90, marker="o", facecolors="none",
              edgecolors=BLUE, linewidths=2, zorder=5, label="Ceros")

# Polos: agrupar duplicados en z=0
unique_poles, counts = np.unique(np.round(poles, 6), return_counts=True)
for p, c in zip(unique_poles, counts):
    ax_pz.scatter([p.real], [p.imag], s=120, marker="x",
                  color=RED, linewidths=2.5, zorder=5,
                  label=f"Polo (×{c})" if c > 1 else "Polo")

ax_pz.set_xlim(-1.6, 1.6)
ax_pz.set_ylim(-1.6, 1.6)
ax_pz.legend(fontsize=8, facecolor=DARK, edgecolor=GRID,
             labelcolor=TEXT, loc="lower right")

# ── Módulo ────────────────────────────────────
ax_mag.set_title("Respuesta de MÓDULO", fontsize=11)
ax_mag.set_xlabel(r"$\omega$ [rad/muestra]")
ax_mag.set_ylabel("|T| [dB]")
ax_mag.plot(w, mag_dB, color=GREEN, linewidth=1.8)
ax_mag.set_xlim(0, np.pi)
ax_mag.set_ylim(bottom=max(mag_dB.min() - 5, -90))
ax_mag.set_xticks(xticks)
ax_mag.set_xticklabels(xlabels)

# ── Fase ──────────────────────────────────────
ax_ph.set_title("Respuesta de FASE", fontsize=11)
ax_ph.set_xlabel(r"$\omega$ [rad/muestra]")
ax_ph.set_ylabel(r"$\angle T$ [rad]")
ax_ph.plot(w, phase_rad, color=BLUE, linewidth=1.8, label="Fase calculada")
ax_ph.plot(w, phase_teo, color=RED, linewidth=1.2, linestyle="--", alpha=0.7,
           label=rf"Fase teórica: $-{retardo}\omega$")
ax_ph.set_xlim(0, np.pi)
ax_ph.set_xticks(xticks)
ax_ph.set_xticklabels(xlabels)
ax_ph.legend(fontsize=9, facecolor=DARK, edgecolor=GRID, labelcolor=TEXT)

# Yticks de fase en unidades de π
fase_min = phase_rad.min()
fase_max = phase_rad.max()
yticks_fase = np.arange(np.floor(fase_min / np.pi), np.ceil(fase_max / np.pi) + 0.5, 0.5) * np.pi
ylabels_fase = []
for v in yticks_fase:
    frac = v / np.pi
    if frac == 0:
        ylabels_fase.append("0")
    elif frac == 1:
        ylabels_fase.append("π")
    elif frac == -1:
        ylabels_fase.append("−π")
    elif frac == int(frac):
        ylabels_fase.append(f"{int(frac)}π")
    else:
        from fractions import Fraction
        f = Fraction(frac).limit_denominator(8)
        ylabels_fase.append(rf"$\frac{{{f.numerator}\pi}}{{{f.denominator}}}$")

#ax_ph.set_yticks(tick_vals)
ax_ph.set_yticks(yticks_fase)
ax_ph.set_ylim(fase_min - 0.2, fase_max + 0.2)
ax_ph.set_yticklabels(ylabels_fase)

plt.show()