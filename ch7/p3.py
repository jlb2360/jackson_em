"""
This program plots the fraction of energy in transmitted
versus reflective wave as you change angle and d
for proplem 7.3 in jackson electrodynamics book
"""

import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

def R(alpha, beta):
    return ( ( 1 - alpha**2 )**2 * np.sin(beta)**2 ) / ( 4*alpha**2 * np.cos(beta)**2 + (1 + alpha**2)**2 * np.sin(beta)**2 )

def T(alpha, beta):
    return 4*alpha**2 / ( 4*alpha**2 * np.cos(beta)**2 + (1 + alpha**2)**2 * np.sin(beta)**2 )

def alpha(n, i):
    return np.sqrt(1 - n**2 * np.sin(i)**2)/(n*np.cos(i))

def beta(n, i, d, fc):
    return (fc * d / c) / np.sqrt(1 - n**2 * np.sin(i)**2)


# define the units which define the scenario
d = 0.1 # 1 unitless
n = 1.5
fc = 10E9

i_tot_refl = np.arcsin(1/n)

i_range = np.linspace(0.00001, i_tot_refl-0.01, 2000)

alphas = alpha(n, i_range)
betas = beta(n, i_range, d, fc)

R_coeffs = R(alphas, betas)
T_coeffs = T(alphas, betas)


print(f"The angle of total reflection is {np.rad2deg(i_tot_refl)}")

plt.plot(np.rad2deg(i_range), R_coeffs*100, label="Reflection Coefficient")
plt.plot(np.rad2deg(i_range), T_coeffs*100, label="Transmission Coefficient")
plt.title(f"Wave Transmission with Carrier Frequency {fc/1E9} GHz")
plt.ylabel("Fraction of Energy [%]")
plt.xlabel("Angle of Incidence [degree]")
plt.legend()
plt.savefig("plots/p3.png")
