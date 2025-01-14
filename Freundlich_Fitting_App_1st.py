import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox


def freundlich_adsorption(C, KF, n):
    """Freundlich adsorption model equation."""
    return KF * (C ** n)


def analyze_freundlich(C, S):
    """
    Analyzes adsorption data using the Freundlich model.

    Args:
        C (list or pandas.Series): List or series of corresponding adsorption amounts.
        S (list or pandas.Series): List or series of corresponding adsorption amounts.

    Returns:
        tuple: A tuple containing:
            - fitted_params (list): List of fitted parameters [KF, n].
            - r_squared (float): R-squared value.
            - fig (matplotlib.figure.Figure): The generated figure of data and model fitting.
    """
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    # Curve fitting using scipy.optimize.curve_fit
    try:
        popt, pcov, *rest = curve_fit(freundlich_adsorption, C, S, p0=[1.0, 1.0])
    except RuntimeError:
        raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    KF_fit, n_fit = popt
    fitted_params = [KF_fit, n_fit]

    # Generate smooth curve fit data
    C_smooth = np.linspace(min(C), max(C), 500)  # Increase the number of points for a smoother curve
    S_fit = freundlich_adsorption(C_smooth, KF_fit, n_fit)

    # Calculate R-squared
    S_fit_original = freundlich_adsorption(C, KF_fit, n_fit)
    r_squared = r2_score(S, S_fit_original)

    return fitted_params, r_squared, C_smooth, S_fit


def add_data():
    """Adds data from the entry fields to the data list."""
    c_value = c_entry.get().strip()
    s_value = s_entry.get().strip()
    if c_value and s_value:
        try:
            c = float(c_value)
            s = float(s_value)
            item_id = len(data_list)
            data_list.append((c, s))
            check_vars[item_id] = tk.BooleanVar(value=True)
            c_entry.delete(0, 'end')
            s_entry.delete(0, 'end')
            refresh_data_frame()  # Refresh the data display
        except ValueError:
            messagebox.showerror("Error", "Invalid input, C and S values must be numeric.")
    else:
        messagebox.showerror("Error", "Please enter both C and S values.")


def delete_data():
    """Deletes selected data from the data list."""
    selected_items = [i for i, var in check_vars.items() if var.get()]
    if not selected_items:
        messagebox.showerror("Error", "Please select data to delete.")
        return
    for item in selected_items:
        del check_vars[item]
    global data_list
    data_list = [data for i, data in enumerate(data_list) if i not in selected_items]
    refresh_data_frame()  # Refresh the data display


def analyze_data():
    """Analyzes data from the data list and displays the results."""
    selected_data = [data for i, data in enumerate(data_list) if check_vars[i].get()]

    if not selected_data:
        messagebox.showerror("Error", "Please select data to analyze.")
        return

    C_values = [data[0] for data in selected_data]
    S_values = [data[1] for data in selected_data]

    all_C_values = [data[0] for data in data_list]
    all_S_values = [data[1] for data in data_list]

    try:
        fitted_params, r_squared, C_smooth, S_fit = analyze_freundlich(C_values, S_values)
        KF_fit, n_fit = fitted_params
        result_text.config(state=tk.NORMAL)
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END,
                           f"Fitted Freundlich Equation: S = ({KF_fit:.3f} * C ^ {n_fit:.3f})\n")
        result_text.insert(tk.END, f"R-squared: {r_squared:.3f}")
        result_text.config(state=tk.DISABLED)

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(all_C_values, all_S_values, label='All Data Points', color='blue')
        ax.scatter(C_values, S_values, label='Selected Data Points', color='green')
        ax.plot(C_smooth, S_fit, 'r-', label='Freundlich Fit')
        ax.set_xlabel('Concentration (C)')
        ax.set_ylabel('Adsorption Amount (S)')
        ax.set_title('Freundlich Adsorption Isotherm')
        ax.legend()
        ax.grid(True)

        # Set the x-axis and y-axis range to include all data points and smooth curve
        ax.set_xlim(0, max(all_C_values) * 1.1)
        ax.set_ylim(0, max(all_S_values) * 1.1)

        # Embed plot into tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=3, rowspan=8, padx=10, pady=10, sticky='ns')
        canvas.draw()
    except RuntimeError as e:
        messagebox.showerror("Error", f"Analysis failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def refresh_data_frame():
    """Refreshes the data display frame."""
    for widget in data_frame.winfo_children():
        widget.destroy()
    for i, (c, s) in enumerate(data_list):
        tk.Label(data_frame, text=f"{c:.4f}").grid(row=i, column=0, padx=5, pady=2)
        tk.Label(data_frame, text=f"{s:.4f}").grid(row=i, column=1, padx=5, pady=2)
        tk.Checkbutton(data_frame, variable=check_vars[i]).grid(row=i, column=2, padx=5, pady=2)


# GUI setup
window = tk.Tk()
window.title("Freundlich Adsorption Analysis")

# Input fields
tk.Label(window, text="Concentration (C):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
c_entry = tk.Entry(window)
c_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
tk.Label(window, text="Adsorption (S):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
s_entry = tk.Entry(window)
s_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
add_button = tk.Button(window, text="Add Data", command=add_data)
add_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')

# Delete Data button
delete_button = tk.Button(window, text="Delete Data", command=delete_data)
delete_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')

# Data display frame
data_frame = tk.Frame(window)
data_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
data_list = []
check_vars = {}

# Analysis button
analyze_button = tk.Button(window, text="Analyze Data", command=analyze_data)
analyze_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# Result display
result_text = tk.Text(window, height=4, width=60, state=tk.DISABLED)
result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# Configure grid weights to allow for resizing
window.grid_columnconfigure(3, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_rowconfigure(5, weight=1)

window.mainloop()