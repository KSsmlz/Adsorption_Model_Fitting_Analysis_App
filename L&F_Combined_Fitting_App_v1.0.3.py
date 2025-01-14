import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
try:
    from ttkthemes import ThemedTk
except ImportError:
    print("Please install ttkthemes: pip install ttkthemes")
    exit()

# Langmuir adsorption model equation
def langmuir_adsorption(C, Smax, K):
    return (Smax * K * C) / (1 + K * C)

# Freundlich adsorption model equation
def freundlich_adsorption(C, KF, n):
    return KF * (C ** n)

# Analyzes adsorption data using the Langmuir model
def analyze_langmuir(C, S):
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    try:
        popt, pcov, *rest = curve_fit(langmuir_adsorption, C, S, p0=[max(S), 1.0])
    except RuntimeError:
        raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    Smax_fit, K_fit = popt
    fitted_params = [Smax_fit, K_fit]

    C_smooth = np.linspace(min(C), max(C), 500)
    S_fit = langmuir_adsorption(C_smooth, Smax_fit, K_fit)

    S_fit_original = langmuir_adsorption(C, Smax_fit, K_fit)
    r_squared = r2_score(S, S_fit_original)

    return fitted_params, r_squared, C_smooth, S_fit

# Analyzes adsorption data using the Freundlich model
def analyze_freundlich(C, S):
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    try:
        popt, pcov, *rest = curve_fit(freundlich_adsorption, C, S, p0=[1.0, 1.0])
    except RuntimeError:
        raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    KF_fit, n_fit = popt
    fitted_params = [KF_fit, n_fit]

    C_smooth = np.linspace(min(C), max(C), 500)
    S_fit = freundlich_adsorption(C_smooth, KF_fit, n_fit)

    S_fit_original = freundlich_adsorption(C, KF_fit, n_fit)
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

    model_type = model_var.get()

    try:
        if model_type == "Langmuir":
            fitted_params, r_squared, C_smooth, S_fit = analyze_langmuir(C_values, S_values)
            Smax_fit, K_fit = fitted_params
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END,
                               f"Fitted Langmuir Equation: S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)\n")
            result_text.insert(tk.END, f"R-squared: {r_squared:.3f}")
            result_text.config(state=tk.DISABLED)

            # Create subplots for both regular and C/S vs. C plots
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi = 100)  # Adjust figure size and DPI
            ax1.scatter(all_C_values, all_S_values, label='All Data Points', color='#66b3ff', alpha=0.5, s=15)
            ax1.scatter(C_values, S_values, label='Selected Data Points', color='#008000', s = 20)
            ax1.plot(C_smooth, S_fit, 'r-', label='Langmuir Fit', linewidth=1.5)
            ax1.set_xlabel('Concentration (C)', fontsize=10)
            ax1.set_ylabel('Adsorption Amount (S)', fontsize=10)
            ax1.set_title('Langmuir Adsorption Isotherm', fontsize=12, fontweight = 'bold')
            ax1.legend(fontsize=8, frameon = True, facecolor = 'white')
            ax1.grid(True, linestyle='--', alpha=0.6)


            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi = 100)  # Adjust figure size and DPI
            ax2.scatter(all_C_values, np.array(all_C_values) / np.array(all_S_values), label='All Data Points',
                        color='#66b3ff', alpha=0.5, s = 15)
            ax2.scatter(C_values, np.array(C_values) / np.array(S_values), label='Selected Data Points', color='#ffa500', s = 20)

            # Fitting curve for C/S vs C
            C_fit = np.array(C_values)
            CS_fit = np.array(C_values) / np.array(S_values)
            p_fit = np.polyfit(C_fit, CS_fit, 1)  # Linear fit for C/S vs C
            C_fit_smooth = np.linspace(min(C_fit), max(C_fit), 500)
            CS_fit_smooth = np.polyval(p_fit, C_fit_smooth)
            ax2.plot(C_fit_smooth, CS_fit_smooth, 'r--', label='Fit', linewidth=1.5)

            ax2.set_xlabel('Concentration (C)', fontsize=10)
            ax2.set_ylabel('C/S', fontsize=10)
            ax2.set_title('Langmuir C/S vs. C Plot', fontsize=12, fontweight = 'bold')
            ax2.legend(fontsize=8, frameon = True, facecolor = 'white')
            ax2.grid(True, linestyle='--', alpha=0.6)

            # Create a notebook for tabbed plot viewing
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=3, rowspan=9, padx=10, pady=10, sticky='ns')

            # Create frames to hold the canvas widgets
            frame1 = ttk.Frame(notebook)
            frame2 = ttk.Frame(notebook)
            notebook.add(frame1, text='Langmuir Isotherm')
            notebook.add(frame2, text='Langmuir C/S vs. C')

            # Embed plots into the frames
            canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
            canvas_widget1 = canvas1.get_tk_widget()
            canvas_widget1.pack(fill=tk.BOTH, expand=True)
            canvas1.draw()

            canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.pack(fill=tk.BOTH, expand=True)
            canvas2.draw()

        elif model_type == "Freundlich":
            fitted_params, r_squared, C_smooth, S_fit = analyze_freundlich(C_values, S_values)
            KF_fit, n_fit = fitted_params
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, f"Fitted Freundlich Equation: S = ({KF_fit:.3f} * C ^ {n_fit:.3f})\n")
            result_text.insert(tk.END, f"R-squared: {r_squared:.3f}")
            result_text.config(state=tk.DISABLED)

            # Create subplots for both regular and log scale plots
            fig3, ax3 = plt.subplots(figsize=(6, 4), dpi = 100)  # Adjust figure size and DPI
            ax3.scatter(all_C_values, all_S_values, label='All Data Points', color='#66b3ff', alpha = 0.5, s=15)
            ax3.scatter(C_values, S_values, label='Selected Data Points', color='#008000', s = 20)
            ax3.plot(C_smooth, S_fit, 'r-', label='Freundlich Fit', linewidth=1.5)
            ax3.set_xlabel('Concentration (C)', fontsize=10)
            ax3.set_ylabel('Adsorption Amount (S)', fontsize=10)
            ax3.set_title('Freundlich Adsorption Isotherm', fontsize=12, fontweight = 'bold')
            ax3.legend(fontsize=8, frameon = True, facecolor = 'white')
            ax3.grid(True, linestyle='--', alpha=0.6)


            fig4, ax4 = plt.subplots(figsize=(6, 4), dpi = 100)  # Adjust figure size and DPI
            ax4.scatter(np.log10(all_C_values), np.log10(all_S_values), label = 'All Data Points', color = '#66b3ff', alpha = 0.5, s = 15)
            ax4.scatter(np.log10(C_values), np.log10(S_values), label='Selected Data Points', color='#ffa500', s = 20)
            ax4.plot(np.log10(C_smooth), np.log10(S_fit), 'r--', label='Log Scale Fit', linewidth=1.5)
            ax4.set_xlabel('log(Concentration (C))', fontsize=10)
            ax4.set_ylabel('log(Adsorption Amount (S))', fontsize=10)
            ax4.set_title('Log Scale Freundlich Adsorption Isotherm', fontsize=12, fontweight = 'bold')
            ax4.legend(fontsize=8, frameon = True, facecolor = 'white')
            ax4.grid(True, linestyle='--', alpha=0.6)

            # Create a notebook for tabbed plot viewing
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=3, rowspan=9, padx=10, pady=10, sticky='ns')

            # Create frames to hold the canvas widgets
            frame3 = ttk.Frame(notebook)
            frame4 = ttk.Frame(notebook)
            notebook.add(frame3, text='Freundlich Isotherm')
            notebook.add(frame4, text='Freundlich Log Scale')

            # Embed plots into the frames
            canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
            canvas_widget3 = canvas3.get_tk_widget()
            canvas_widget3.pack(fill=tk.BOTH, expand=True)
            canvas3.draw()

            canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
            canvas_widget4 = canvas4.get_tk_widget()
            canvas_widget4.pack(fill=tk.BOTH, expand=True)
            canvas4.draw()
        else:
             messagebox.showerror("Error", "Please select a valid model (Langmuir or Freundlich).")
    except RuntimeError as e:
        messagebox.showerror("Error", f"Analysis failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Refreshes the data display frame
def refresh_data_frame():
    for widget in data_frame.winfo_children():
        widget.destroy()
    for i, (c, s) in enumerate(data_list):
        tk.Label(data_frame, text=f"{c:.4f}", font = ('Arial', 10), bg = '#f0f0f0').grid(row=i, column=0, padx=5, pady=2, sticky = 'w')
        tk.Label(data_frame, text=f"{s:.4f}", font = ('Arial', 10), bg = '#f0f0f0').grid(row=i, column=1, padx=5, pady=2, sticky = 'w')
        tk.Checkbutton(data_frame, variable=check_vars[i], bg = '#f0f0f0').grid(row=i, column=2, padx=5, pady=2, sticky = 'w')

# GUI setup
window = ThemedTk(theme="xpnative")
window.title("Adsorption Analysis")
window.configure(background = 'white')
style = ttk.Style(window)
style.configure('TButton', font = ('Arial', 12), borderwidth = 0, relief = 'flat', background = '#4CAF50', foreground = 'white', padding = 8)
# Style for buttons
button_style = {
    'font': ('Arial', 12),
    'borderwidth': 0,
    'relief': 'flat',
    'background': '#4CAF50',
    'foreground': 'white',
}

# Style for labels
label_style = {
    'font': ('Arial', 12),
    'background': 'white',
    'foreground': '#333333'
}

# Style for entry
entry_style = {
    'font': ('Arial', 12),
    'borderwidth': 1,
    'relief': 'solid',
    'highlightthickness': 0
}


# Style for combobox
combobox_style = {
    'font': ('Arial', 12),
    'borderwidth': 0, # Removed borderwidth
    'relief': 'solid',
    'highlightthickness': 0,
    'background': 'white'
}

# Model selection dropdown
tk.Label(window, text="Select Model:", **label_style).grid(row=0, column=0, padx=5, pady=5, sticky='e')
model_var = tk.StringVar(window)
model_var.set("Langmuir")  # Default value
model_dropdown = ttk.Combobox(window, textvariable=model_var, values=["Langmuir", "Freundlich"], font = ('Arial', 12)) # apply font here
model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky='w')


tk.Label(window, text="Concentration (C):", **label_style).grid(row=1, column=0, padx=5, pady=5, sticky='e')
c_entry = tk.Entry(window, **entry_style)
c_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

tk.Label(window, text="Adsorption (S):", **label_style).grid(row=2, column=0, padx=5, pady=5, sticky='e')
s_entry = tk.Entry(window, **entry_style)
s_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

add_button = ttk.Button(window, text="Add Data", command=add_data, style = 'TButton')
add_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')

delete_button = ttk.Button(window, text="Delete Data", command=delete_data, style = 'TButton')
delete_button.grid(row=3, column=2, padx=5, pady=5, sticky='w')

data_frame = tk.Frame(window, bg = '#f0f0f0')
data_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
data_list = []
check_vars = {}


analyze_button = ttk.Button(window, text="Analyze Data", command=analyze_data, style = 'TButton')
analyze_button.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

result_text = tk.Text(window, height=4, width=60, state=tk.DISABLED, font = ('Arial', 12), bg = '#f0f0f0')
result_text.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

window.grid_columnconfigure(3, weight=1)
window.grid_rowconfigure(4, weight=1)
window.grid_rowconfigure(6, weight=1)


window.mainloop()