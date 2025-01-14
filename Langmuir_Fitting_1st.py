import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def langmuir_adsorption(C, Smax, K):
    """Langmuir adsorption model equation."""
    return (Smax * K * C) / (1 + K * C)

def analyze_langmuir(C, S):
    """
    Analyzes adsorption data using the Langmuir model.

    Args:
        C (list or pandas.Series): List or series of concentrations.
        S (list or pandas.Series): List or series of corresponding adsorption amounts.

    Returns:
        tuple: A tuple containing:
            - fitted_params (list): List of fitted parameters [Smax, K].
            - r_squared (float): R-squared value.
            - fig (matplotlib.figure.Figure): The generated figure of data and model fitting.
    """
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    # Curve fitting using scipy.optimize.curve_fit
    try:
        popt, pcov, *rest = curve_fit(langmuir_adsorption, C, S, p0=[max(S), 1.0])
    except RuntimeError:
         raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    Smax_fit, K_fit = popt
    fitted_params = [Smax_fit, K_fit]

    # Generate curve fit data
    S_fit = langmuir_adsorption(C, Smax_fit, K_fit)

    # Calculate R-squared
    r_squared = r2_score(S, S_fit)


    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(C, S, label='Experimental Data')
    ax.plot(C, S_fit, 'r-', label='Langmuir Fit')
    ax.set_xlabel('Concentration (C)')
    ax.set_ylabel('Adsorption Amount (S)')
    ax.set_title('Langmuir Adsorption Isotherm')
    ax.legend()
    ax.grid(True)

    ax.set_xlim(0, max(C) * 1.1)
    ax.set_ylim(0, max(S) * 1.1)

    return fitted_params, r_squared, fig

if __name__ == '__main__':
    # Example usage (you can replace this with actual user input)
    data = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'S': [1.2, 1.9, 2.5, 3.1, 3.6, 4.0, 4.3, 4.6, 4.8, 5.0]}
    df = pd.DataFrame(data)

    try:
      fitted_params, r_squared, fig = analyze_langmuir(df['C'], df['S'])

      Smax_fit, K_fit = fitted_params

      print(f"Fitted Langmuir Equation: S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)")
      print(f"R-squared: {r_squared:.3f}")

      plt.show()
    except RuntimeError as e:
        print(f"Error: {e}")