import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk


# Langmuir adsorption model equation
def langmuir_adsorption(C, Smax, K):
    return (Smax * K * C) / (1 + K * C)


# Analyzes adsorption data using the Langmuir model
def analyze_langmuir(C, S):
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    try:
        popt, pcov = curve_fit(langmuir_adsorption, C, S, p0=[max(S), 1.0])
    except RuntimeError:
        raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    Smax_fit, K_fit = popt
    fitted_params = [Smax_fit, K_fit]

    C_smooth = np.linspace(min(C), max(C), 500)
    S_fit = langmuir_adsorption(C_smooth, Smax_fit, K_fit)

    S_fit_original = langmuir_adsorption(C, Smax_fit, K_fit)
    r_squared = r2_score(S, S_fit_original)

    return fitted_params, r_squared, C_smooth, S_fit


# Adds data from the entry fields to the data list
def add_data():
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
            refresh_data_frame()
        except ValueError:
            messagebox.showerror("Error", "Invalid input, C and S values must be numeric.")
    else:
        messagebox.showerror("Error", "Please enter both C and S values.")


# Deletes selected data from the data list
def delete_data():
    selected_items = [i for i, var in check_vars.items() if var.get()]
    if not selected_items:
        messagebox.showerror("Error", "Please select data to delete.")
        return
    for item in selected_items:
        del check_vars[item]
    global data_list
    data_list = [data for i, data in enumerate(data_list) if i not in selected_items]
    refresh_data_frame()


# Analyzes data from the data list and displays the results
def analyze_data():
    selected_data = [data for i, data in enumerate(data_list) if check_vars[i].get()]

    if not selected_data:
        messagebox.showerror("Error", "Please select data to analyze.")
        return

    C_values = [data[0] for data in selected_data]
    S_values = [data[1] for data in selected_data]

    all_C_values = [data[0] for data in data_list]
    all_S_values = [data[1] for data in data_list]

    try:
        fitted_params, r_squared, C_smooth, S_fit = analyze_langmuir(C_values, S_values)
        Smax_fit, K_fit = fitted_params
        result_text.config(state=tk.NORMAL)
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END,
                           f"Fitted Langmuir Equation: S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)\n")
        result_text.insert(tk.END, f"R-squared: {r_squared:.3f}")
        result_text.config(state=tk.DISABLED)

        # Create subplots for both regular and C/S vs. C plots
        fig1, ax1 = plt.subplots()
        ax1.scatter(all_C_values, all_S_values, label='All Data Points', color='blue')
        ax1.scatter(C_values, S_values, label='Selected Data Points', color='green')
        ax1.plot(C_smooth, S_fit, 'r-', label='Langmuir Fit')
        ax1.set_xlabel('Concentration (C)')
        ax1.set_ylabel('Adsorption Amount (S)')
        ax1.set_title('Langmuir Adsorption Isotherm')
        ax1.legend()
        ax1.grid(True)

        fig2, ax2 = plt.subplots()
        ax2.scatter(all_C_values, np.array(all_C_values) / np.array(all_S_values), label='All Data Points',
                    color='blue')
        ax2.scatter(C_values, np.array(C_values) / np.array(S_values), label='Selected Data Points', color='orange')

        # Fitting curve for C/S vs C
        C_fit = np.array(C_values)
        CS_fit = np.array(C_values) / np.array(S_values)
        p_fit = np.polyfit(C_fit, CS_fit, 1)  # Linear fit for C/S vs C
        C_fit_smooth = np.linspace(min(C_fit), max(C_fit), 500)
        CS_fit_smooth = np.polyval(p_fit, C_fit_smooth)
        ax2.plot(C_fit_smooth, CS_fit_smooth, 'r--', label='Fit')

        ax2.set_xlabel('Concentration (C)')
        ax2.set_ylabel('C/S')
        ax2.set_title('C/S vs. C Plot')
        ax2.legend()
        ax2.grid(True)

        # Create a notebook for tabbed plot viewing
        notebook = ttk.Notebook(window)
        notebook.grid(row=0, column=3, rowspan=8, padx=10, pady=10, sticky='ns')

        # Create frames to hold the canvas widgets
        frame1 = ttk.Frame(notebook)
        frame2 = ttk.Frame(notebook)
        notebook.add(frame1, text='Langmuir Isotherm')
        notebook.add(frame2, text='C/S vs. C')

        # Embed plots into the frames
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.pack(fill=tk.BOTH, expand=True)
        canvas1.draw()

        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.pack(fill=tk.BOTH, expand=True)
        canvas2.draw()
    except RuntimeError as e:
        messagebox.showerror("Error", f"Analysis failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


# Refreshes the data display frame
def refresh_data_frame():
    for widget in data_frame.winfo_children():
        widget.destroy()
    for i, (c, s) in enumerate(data_list):
        tk.Label(data_frame, text=f"{c:.4f}").grid(row=i, column=0, padx=5, pady=2)
        tk.Label(data_frame, text=f"{s:.4f}").grid(row=i, column=1, padx=5, pady=2)
        tk.Checkbutton(data_frame, variable=check_vars[i]).grid(row=i, column=2, padx=5, pady=2)


# GUI setup
window = tk.Tk()
window.title("Langmuir Adsorption Analysis")

tk.Label(window, text="Concentration (C):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
c_entry = tk.Entry(window)
c_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
tk.Label(window, text="Adsorption (S):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
s_entry = tk.Entry(window)
s_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
add_button = tk.Button(window, text="Add Data", command=add_data)
add_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')

delete_button = tk.Button(window, text="Delete Data", command=delete_data)
delete_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')

data_frame = tk.Frame(window)
data_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
data_list = []
check_vars = {}

analyze_button = tk.Button(window, text="Analyze Data", command=analyze_data)
analyze_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

result_text = tk.Text(window, height=4, width=60, state=tk.DISABLED)
result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

window.grid_columnconfigure(3, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_rowconfigure(5, weight=1)

window.mainloop()
