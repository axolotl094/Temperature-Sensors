import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

d = np.load('data.npz')
T_cal, R_cal = d['T_cal'], d['R_cal']
R, tc_mV, sd_V, T_C, T_K = d['R'], d['tc_mV'], d['sd_V'], d['T_C'], d['T_K']

def linfit(x, y):
    n = len(x); xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x-xm)**2); Sxy = np.sum((x-xm)*(y-ym))
    b = Sxy/Sxx; a = ym - b*xm
    return a, b

# ---- Plot 1: Pt-100 calibration ----
a_cal, b_cal = linfit(T_cal, R_cal)
Tline = np.linspace(-210, 105, 200)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(T_cal, R_cal, 'o', color='crimson', ms=8, label='Calibration data', zorder=3)
ax.plot(Tline, a_cal + b_cal*Tline, '-', color='navy', lw=1.6,
        label=fr'Fit: $R = {a_cal:.2f} + {b_cal:.4f}\,T$')
ax.set_xlabel(r'Temperature $T$ (°C)')
ax.set_ylabel(r'Resistance $R$ ($\Omega$)')
ax.set_title('Calibration of Pt-100 resistance thermometer')
ax.legend()
fig.tight_layout()
fig.savefig('pt100_calibration.pdf')
plt.close(fig)

# ---- Plot 2: Diode V vs T(K) ----
a_sd, b_sd = linfit(T_K, sd_V)
Tk_line = np.linspace(70, 300, 200)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(T_K, sd_V, 'o', color='darkorange', ms=6, label='Data', zorder=3)
ax.plot(Tk_line, a_sd + b_sd*Tk_line, '-', color='navy', lw=1.6,
        label=fr'Fit: $V = {a_sd:.4f} {b_sd:.6f}\,T$')
ax.set_xlabel(r'Temperature $T$ (K)')
ax.set_ylabel(r'Diode forward voltage $V$ (V)')
ax.set_title('Semiconductor diode (1N4148): $V$ vs $T$')
ax.legend()
fig.tight_layout()
fig.savefig('diode_calibration.pdf')
plt.close(fig)

# ---- Plot 3: Thermocouple EMF vs T(C) ----
a_tc, b_tc = linfit(T_C, tc_mV)
Tc_line = np.linspace(-210, 30, 200)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(T_C, tc_mV, 'o', color='seagreen', ms=6, label='Data', zorder=3)
ax.plot(Tc_line, a_tc + b_tc*Tc_line, '-', color='navy', lw=1.6,
        label=fr'Fit: $\varepsilon = {a_tc:.3f} {b_tc:.5f}\,T$')
ax.set_xlabel(r'Temperature $T$ (°C)')
ax.set_ylabel(r'Thermo-emf $\varepsilon$ (mV)')
ax.set_title('Thermocouple: emf vs $T$')
ax.legend()
fig.tight_layout()
fig.savefig('thermocouple_calibration.pdf')
plt.close(fig)

# ---- Plot 4: Pt-100 calibration residuals ----
resid = R_cal - (a_cal + b_cal*T_cal)
fig, ax = plt.subplots(figsize=(6,3.2))
ax.axhline(0, color='k', lw=1)
ax.stem(T_cal, resid, basefmt=' ')
ax.set_xlabel(r'Temperature $T$ (°C)')
ax.set_ylabel(r'Residual $R_{obs}-R_{fit}$ ($\Omega$)')
ax.set_title('Residuals of Pt-100 calibration fit')
fig.tight_layout()
fig.savefig('pt100_residuals.pdf')
plt.close(fig)

print("Plots written.")
print("Pt100 fit:", a_cal, b_cal)
print("Diode fit:", a_sd, b_sd)
print("TC fit:", a_tc, b_tc)
