import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

def langmuir_adsorption(C, Smax, K):
    """Langmuir adsorption model equation."""
    return (Smax * K * C) / (1 + K * C)

def analyze_langmuir(C, S):
    """
    Analyzes adsorption data using the Langmuir model.

    Args:
        C (list or pandas.Series): List or series of corresponding adsorption amounts.
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

def add_data():
    """Adds data from the entry fields to the Treeview."""
    c_value = c_entry.get().strip()
    s_value = s_entry.get().strip()
    if c_value and s_value:
        try:
          c = float(c_value)
          s = float(s_value)
          data_tree.insert('', 'end', values=(c, s))
          c_entry.delete(0, 'end')
          s_entry.delete(0, 'end')
        except ValueError:
          messagebox.showerror("Error", "Invalid input, C and S values must be numeric.")
    else:
        messagebox.showerror("Error", "Please enter both C and S values.")
def analyze_data():
  """Analyzes data from the Treeview and displays the results."""
  data = data_tree.get_children()
  if not data:
      messagebox.showerror("Error", "Please enter data first.")
      return

  C_values = []
  S_values = []
  for item in data:
    c, s = data_tree.item(item, 'values')
    C_values.append(float(c))
    S_values.append(float(s))
  try:
    fitted_params, r_squared, fig = analyze_langmuir(C_values, S_values)
    Smax_fit, K_fit = fitted_params
    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, f"Fitted Langmuir Equation: S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)\n")
    result_text.insert(tk.END, f"R-squared: {r_squared:.3f}")
    result_text.config(state=tk.DISABLED)
    # Embed plot into tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=3, rowspan=7, padx=10, pady=10, sticky='ns') # Modified here
  except RuntimeError as e:
      messagebox.showerror("Error", f"Analysis failed: {e}")
  except Exception as e:
    messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# GUI setup
window = tk.Tk()
window.title("Langmuir Adsorption Analysis")

# Input fields
tk.Label(window, text="Concentration (C):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
c_entry = tk.Entry(window)
c_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
tk.Label(window, text="Adsorption (S):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
s_entry = tk.Entry(window)
s_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
add_button = tk.Button(window, text="Add Data", command=add_data)
add_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')

# Data display
data_tree = ttk.Treeview(window, columns=('C', 'S'), show='headings')
data_tree.heading('C', text='Concentration (C)')
data_tree.heading('S', text='Adsorption Amount (S)')
data_tree.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# Analysis button
analyze_button = tk.Button(window, text="Analyze Data", command=analyze_data)
analyze_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# Result display
result_text = tk.Text(window, height=4, width=60, state=tk.DISABLED)
result_text.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# Configure grid weights to allow for resizing
window.grid_columnconfigure(3, weight=1)  # Added this to allow the plot to expand
window.grid_rowconfigure(2, weight=1) # Added this to allow the data_tree to expand
window.grid_rowconfigure(4, weight=1) # Added this to allow the text area to expand

window.mainloop()