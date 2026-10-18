import numpy as np

# ---------------- 1) Pt-100 calibration data ----------------
T_cal = np.array([24.8, 96.1, 0.5, -196.0])   # deg C
R_cal = np.array([110.1, 136.2, 100.8, 20.0]) # ohm

def linfit(x, y):
    n = len(x)
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x-xm)**2)
    Sxy = np.sum((x-xm)*(y-ym))
    b = Sxy/Sxx
    a = ym - b*xm
    yfit = a + b*x
    resid = y - yfit
    SSR = np.sum(resid**2)
    s = np.sqrt(SSR/(n-2)) if n>2 else np.nan
    db = s/np.sqrt(Sxx) if n>2 else np.nan
    da = s*np.sqrt(np.sum(x**2)/(n*Sxx)) if n>2 else np.nan
    Syy = np.sum((y-ym)**2)
    r = Sxy/np.sqrt(Sxx*Syy)
    return dict(a=a,b=b,da=da,db=db,s=s,r=r,resid=resid,n=n)

cal = linfit(T_cal, R_cal)
print("=== Pt-100 calibration: R = a + b*T ===")
print(f"a (R0) = {cal['a']:.4f} +/- {cal['da']:.4f} ohm")
print(f"b (alpha*R0) = {cal['b']:.5f} +/- {cal['db']:.5f} ohm/C")
print(f"alpha = b/a = {cal['b']/cal['a']*1000:.4f} x1e-3 /C")
print(f"r = {cal['r']:.5f}, r^2={cal['r']**2:.5f}")
print(f"std err of fit s = {cal['s']:.4f} ohm")
print(f"residuals: {cal['resid']}")
print()

a_cal, b_cal = cal['a'], cal['b']
da_cal, db_cal = cal['da'], cal['db']

# ---------------- 2) main dataset ----------------
R = np.array([20.0,29.2,30.2,35.0,40.0,45.9,50.1,55.0,60.0,65.0,70.0,75.0,80.0,85.2,90.0,96.4,100.0,105.0,108.6])
tc_mV = np.array([6.4,6.3,6.3,6.1,5.9,5.7,5.4,5.1,4.7,4.3,3.9,3.5,3.1,2.6,2.2,1.5,1.1,0.6,0.1])
sd_V  = np.array([1.02,1.001,1.01,1.00,0.99,0.97,0.95,0.93,0.91,0.89,0.87,0.85,0.83,0.82,0.80,0.77,0.76,0.74,0.72])

T_C = (R - a_cal)/b_cal
T_K = T_C + 273.15

print("=== Converted temperatures from Pt-100 resistance ===")
for r_,t_ in zip(R,T_C):
    print(f"R={r_:7.2f} ohm  ->  T = {t_:8.2f} C  = {t_+273.15:7.2f} K")
print()

# thermocouple fit: EMF(mV) = c + d*T(C)
tc_fit = linfit(T_C, tc_mV)
print("=== Thermocouple: EMF(mV) = c + d*T(C) ===")
print(f"c = {tc_fit['a']:.4f} +/- {tc_fit['da']:.4f} mV")
print(f"d (Seebeck coeff) = {tc_fit['b']:.6f} +/- {tc_fit['db']:.6f} mV/C = {tc_fit['b']*1000:.3f} +/- {tc_fit['db']*1000:.3f} uV/C")
print(f"r = {tc_fit['r']:.5f}, r^2 = {tc_fit['r']**2:.5f}")
print(f"s = {tc_fit['s']:.4f} mV")
print()

# semiconductor diode fit: V = A - B*T(K) i.e. V = A + B'*T with B'=-B
sd_fit = linfit(T_K, sd_V)
print("=== Diode: V(Volts) = A + B*T(K) ===")
print(f"A (Eg/q, extrapolated to T=0K) = {sd_fit['a']:.5f} +/- {sd_fit['da']:.5f} V")
print(f"B (slope) = {sd_fit['b']:.7f} +/- {sd_fit['db']:.7f} V/K")
print(f"r = {sd_fit['r']:.5f}, r^2 = {sd_fit['r']**2:.5f}")
print(f"s = {sd_fit['s']:.5f} V")
print(f"Estimated band gap energy Eg = q*A = {sd_fit['a']:.3f} eV")
print()

# Save everything needed for LaTeX table / plotting
np.savez('data.npz', T_cal=T_cal, R_cal=R_cal, R=R, tc_mV=tc_mV, sd_V=sd_V, T_C=T_C, T_K=T_K)
