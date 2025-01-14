import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


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

    return fitted_params, r_squared, fig

def get_user_data():
    """Gets C and S data from user input."""
    C_values = []
    S_values = []
    print("Enter C and S values, type 'done' when finished:")
    while True:
        line = input("Enter C value, then S value separated by a comma (e.g., 0.1, 1.2): ").strip().lower()
        if line == 'done':
            break
        try:
            c_str, s_str = line.split(',')
            c = float(c_str.strip())
            s = float(s_str.strip())
            C_values.append(c)
            S_values.append(s)
        except ValueError:
            print("Invalid input format. Please use 'C, S' format, or 'done'.")
    if not C_values:
        return None, None
    return C_values, S_values


if __name__ == '__main__':
    while True:
        print("\nLangmuir Adsorption Analysis Menu:")
        print("1. Enter data")
        print("2. Analyze data")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            C_values, S_values = get_user_data()
            if C_values is not None:
               print("Data entered successfully.")
            else:
               print("No data entered.")
               continue
        elif choice == '2':
             if C_values is None:
                print("Please enter data first.")
                continue
             try:
                fitted_params, r_squared, fig = analyze_langmuir(C_values, S_values)
                Smax_fit, K_fit = fitted_params
                print(f"Fitted Langmuir Equation: S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)")
                print(f"R-squared: {r_squared:.3f}")
                plt.show()
             except RuntimeError as e:
                print(f"Error: {e}")
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")