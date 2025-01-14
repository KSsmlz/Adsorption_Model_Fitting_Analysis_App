import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Langmuir adsorption isotherm model
def langmuir(C, Qmax, KL):
    return (Qmax * KL * C) / (1 + KL * C)

# Freundlich adsorption isotherm model
def freundlich(C, KF, n):
    return KF * C ** (1 / n)

# Experimental data
C = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Equilibrium concentration (mg/L)
Q = np.array([0.9, 1.7, 2.5, 3.2, 3.8, 4.3, 4.8, 5.1, 5.4, 5.6])  # Adsorbed amount (mg/g)

# Fit Langmuir model
popt_langmuir, pcov_langmuir = curve_fit(langmuir, C, Q, bounds=(0, np.inf))
Qmax, KL = popt_langmuir

# Fit Freundlich model
popt_freundlich, pcov_freundlich = curve_fit(freundlich, C, Q, bounds=(0, np.inf))
KF, n = popt_freundlich

# Generate data for plotting
C_plot = np.linspace(0.1, 10, 100)
Q_langmuir = langmuir(C_plot, Qmax, KL)
Q_freundlich = freundlich(C_plot, KF, n)

# Langmuir adsorption isotherm plot
plt.figure(figsize=(10, 6))
plt.scatter(C, Q, label='Experimental Data', color='blue')
plt.plot(C_plot, Q_langmuir, label=f'Langmuir Fit (Qmax={Qmax:.2f}, KL={KL:.2f})', color='red')
plt.title('Langmuir Adsorption Isotherm')
plt.xlabel('Equilibrium Concentration, C (mg/L)')
plt.ylabel('Adsorbed Amount, Q (mg/g)')
plt.legend()
plt.grid()
plt.show()

# C/S vs C plot for Langmuir
CS = C / Q
plt.figure(figsize=(10, 6))
plt.scatter(C, CS, label='Experimental Data', color='green')
plt.plot(C_plot, C_plot / langmuir(C_plot, Qmax, KL), label='Langmuir Fit', color='orange')
plt.title('C/S vs C Plot (Langmuir)')
plt.xlabel('Equilibrium Concentration, C (mg/L)')
plt.ylabel('C/Q (L/mg)')
plt.legend()
plt.grid()
plt.show()

# Freundlich adsorption isotherm plot
plt.figure(figsize=(10, 6))
plt.scatter(C, Q, label='Experimental Data', color='purple')
plt.plot(C_plot, Q_freundlich, label=f'Freundlich Fit (KF={KF:.2f}, 1/n={1/n:.2f})', color='brown')
plt.title('Freundlich Adsorption Isotherm')
plt.xlabel('Equilibrium Concentration, C (mg/L)')
plt.ylabel('Adsorbed Amount, Q (mg/g)')
plt.legend()
plt.grid()
plt.show()

# Log-log plot for Freundlich
plt.figure(figsize=(10, 6))
log_C = np.log10(C)
log_Q = np.log10(Q)
plt.scatter(log_C, log_Q, label='Experimental Data', color='cyan')
plt.plot(np.log10(C_plot), np.log10(Q_freundlich), label='Freundlich Fit', color='magenta')
plt.title('Log-Log Plot (Freundlich Isotherm)')
plt.xlabel('log(C)')
plt.ylabel('log(Q)')
plt.legend()
plt.grid()
plt.show()
