# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from math import gamma
from scipy.optimize import brentq


def f(x):
    return 8*x + np.sqrt(16*x - 64*x**2) - 0.025  # for β <= 0.05


def solve_x(L, C):
    p = L/2 + 1
    return (C / 4 * 2**p * gamma(p+1)) ** (1.0 / p)


if __name__ == '__main__':
    # calculate T for L based on Eq.(S23)
    a, b = 0.0, 0.25
    root = brentq(f, a, b)
    C = root
    L_vals = np.linspace(1, 330, 5000)
    T_vals = np.array([solve_x(L, C) for L in L_vals])

    # calculate L for T ~ 1, 2, 4, 8
    for i, x in enumerate(T_vals):
        if abs(x - 1) < 1e-1:
            print("T ~ 1 when L =", L_vals[i])
        elif abs(x - 2) < 1e-1:
            print("T ~ 2 when L =", L_vals[i])
        elif abs(x - 4) < 1e-1:
            print("T ~ 4 when L =", L_vals[i])
        elif abs(x - 8) < 1e-1:
            print("T ~ 8 when L =", L_vals[i])

    # set fitting range
    T_min, T_max = 10, 100
    mask = (T_vals >= T_min) & (T_vals <= T_max)
    T_sel = T_vals[mask]
    L_sel = L_vals[mask]

    # linear regression
    slope, intercept = np.polyfit(T_sel, L_sel, 1)
    print("slope    : ", slope)
    print("intercept:", intercept)

    L_fit = slope * T_sel + intercept

    # plot T-L 
    plt.figure(figsize=(8, 6))
    plt.plot(T_vals, L_vals, color='blue',
             label=r'$\frac{4T^{L/2 + 1}}{2^{L/2 + 1}(L/2 + 1)!} = 3.813 \times 10^{-5}$', lw=2)
    plt.plot(T_sel, L_fit, 'r--', lw=2, label=f'fit : L = {slope:.2f}T + {intercept:.2f}')
    plt.axvspan(T_min, T_max, color='gray', alpha=0.2)

    plt.xlabel('T', fontsize=18)
    plt.ylabel('L', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(fr"./results/graph_T-L.pdf", dpi=600, bbox_inches='tight')

    plt.close()
