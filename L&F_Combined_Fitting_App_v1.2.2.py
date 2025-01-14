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
import matplotlib.ticker as ticker


# Format y-axis ticks with scientific notation if needed
def format_y_axis(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for tick in ax.get_yticklabels():
        if len(tick.get_text()) >= 5:
            offset = formatter.get_offset()
            tick.set_text(f"{tick.get_text()} × {offset}")


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


group_counter = 0  # Initialize group counter outside the function


# Adds data from the entry fields to the data list
def add_data():
    global group_counter
    c_values_str = c_entry.get().strip()
    s_values_str = s_entry.get().strip()

    if c_values_str and s_values_str:
        try:
            c_values_list = [list(map(float, (c_set.replace(',', ' ').split()))) for c_set in c_values_str.split(' ') if
                             c_set.strip()]
            s_values_list = [list(map(float, (s_set.replace(',', ' ').split()))) for s_set in s_values_str.split(' ') if
                             s_set.strip()]

            if len(c_values_list) != len(s_values_list):
                messagebox.showerror("Error", "Number of C and S value sets do not match.")
                return

            for c_values, s_values in zip(c_values_list, s_values_list):
                if len(c_values) != len(s_values):
                    messagebox.showerror("Error", "Number of C and S values do not match in a data set.")
                    return
                group_counter += 1
                for c, s in zip(c_values, s_values):
                    item_id = len(data_list)
                    data_list.append((c, s, group_counter))  # data_list now contains (C, S, group_id)
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
    group_ids = [data[2] for data in selected_data]

    all_C_values = [data[0] for data in data_list]
    all_S_values = [data[1] for data in data_list]
    all_group_ids = [data[2] for data in data_list]

    model_type = model_var.get()

    try:
        if model_type == "Langmuir":
            fitted_params, r_squared, C_smooth, S_fit = analyze_langmuir(C_values, S_values)
            Smax_fit, K_fit = fitted_params
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, f"Fitted Langmuir Equation:\n")
            result_text.insert(tk.END, f"S = ({Smax_fit:.3f} * {K_fit:.3f} * C) / (1 + {K_fit:.3f} * C)\n")
            result_text.insert(tk.END, f"R-squared: {r_squared:.4f}")
            result_text.config(state=tk.DISABLED)

            # Create subplots for both regular and C/S vs. C plots
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI
            ax1.scatter(all_C_values, all_S_values, label='All Data Points', color='#66b3ff', alpha=0.5, s=15)

            # Calculate standard deviations
            unique_c_values = sorted(list(set(C_values)))
            S_std = []
            for c_val in unique_c_values:
                s_values_for_c = [S_values[i] for i, c in enumerate(C_values) if c == c_val]
                if len(s_values_for_c) > 1:
                    S_std.append(np.std(s_values_for_c))
                else:
                    S_std.append(0)
            ax1.errorbar(unique_c_values,
                         [np.mean([S_values[i] for i, c in enumerate(C_values) if c == c_val]) for c_val in
                          unique_c_values], yerr=S_std, fmt='o', label='Selected Data Points', color='#008000',
                         markersize=5, capsize=5, ecolor='black', elinewidth=0.8)

            ax1.plot(C_smooth, S_fit, 'r-', label='Langmuir Fit', linewidth=1.5)
            ax1.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Adsorption Amount (S) [unit]', fontsize=10, fontweight='bold')
            ax1.set_title('Langmuir Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax1.grid(True, linestyle='--', alpha=0.6)
            fig1.patch.set_facecolor('white')  # Setting background color for the figure

            format_y_axis(ax1)

            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI
            ax2.scatter(all_C_values, np.array(all_C_values) / np.array(all_S_values), label='All Data Points',
                        color='#66b3ff', alpha=0.5, s=15)

            # Calculate standard deviations
            unique_c_values = sorted(list(set(C_values)))
            CS_std = []
            for c_val in unique_c_values:
                cs_values_for_c = [C_values[i] / S_values[i] for i, c in enumerate(C_values) if c == c_val]
                if len(cs_values_for_c) > 1:
                    CS_std.append(np.std(cs_values_for_c))
                else:
                    CS_std.append(0)

            ax2.errorbar(unique_c_values,
                         [np.mean([C_values[i] / S_values[i] for i, c in enumerate(C_values) if c == c_val]) for c_val
                          in unique_c_values], yerr=CS_std, fmt='o', label='Selected Data Points',
                         color='#ffa500', markersize=5, capsize=5, ecolor='black', elinewidth=0.8)

            # Fitting curve for C/S vs C
            C_fit = np.array(C_values)
            CS_fit = np.array(C_values) / np.array(S_values)
            p_fit = np.polyfit(C_fit, CS_fit, 1)  # Linear fit for C/S vs C
            C_fit_smooth = np.linspace(min(C_fit), max(C_fit), 500)
            CS_fit_smooth = np.polyval(p_fit, C_fit_smooth)
            ax2.plot(C_fit_smooth, CS_fit_smooth, 'r--', label='Fit', linewidth=1.5)

            ax2.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax2.set_ylabel('C/S [unit]', fontsize=10, fontweight='bold')
            ax2.set_title('Langmuir C/S vs. C Plot', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax2.grid(True, linestyle='--', alpha=0.6)
            fig2.patch.set_facecolor('white')  # Setting background color for the figure

            format_y_axis(ax2)

            # Create a notebook for tabbed plot viewing
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=3, rowspan=10, padx=(10, 10), pady=10, sticky='ns')

            # Create frames to hold the canvas widgets
            frame1 = ttk.Frame(notebook)
            frame2 = ttk.Frame(notebook)
            notebook.add(frame1, text='Langmuir Isotherm')
            notebook.add(frame2, text='Langmuir C/S vs. C')

            # Embed plots into the frames
            canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
            canvas_widget1 = canvas1.get_tk_widget()
            canvas_widget1.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            canvas1.draw()

            canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            canvas2.draw()

        elif model_type == "Freundlich":
            fitted_params, r_squared, C_smooth, S_fit = analyze_freundlich(C_values, S_values)
            KF_fit, n_fit = fitted_params
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, f"Fitted Freundlich Equation:\n")
            result_text.insert(tk.END, f"S = ({KF_fit:.3f} * C ^ {n_fit:.3f})\n")
            result_text.insert(tk.END, f"R-squared: {r_squared:.4f}")
            result_text.config(state=tk.DISABLED)

            # Create subplots for both regular and log scale plots
            fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI
            ax3.scatter(all_C_values, all_S_values, label='All Data Points', color='#66b3ff', alpha=0.5, s=15)

            # Calculate standard deviations
            unique_c_values = sorted(list(set(C_values)))
            S_std = []
            for c_val in unique_c_values:
                s_values_for_c = [S_values[i] for i, c in enumerate(C_values) if c == c_val]
                if len(s_values_for_c) > 1:
                    S_std.append(np.std(s_values_for_c))
                else:
                    S_std.append(0)
            ax3.errorbar(unique_c_values,
                         [np.mean([S_values[i] for i, c in enumerate(C_values) if c == c_val]) for c_val in
                          unique_c_values], yerr=S_std, fmt='o', label='Selected Data Points', color='#008000',
                         markersize=5, capsize=5, ecolor='black', elinewidth=0.8)

            ax3.plot(C_smooth, S_fit, 'r-', label='Freundlich Fit', linewidth=1.5)
            ax3.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Adsorption Amount (S) [unit]', fontsize=10, fontweight='bold')
            ax3.set_title('Freundlich Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax3.grid(True, linestyle='--', alpha=0.6)
            fig3.patch.set_facecolor('white')

            # Format y-axis ticks with scientific notation if needed
            format_y_axis(ax3)

            fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI

            # Avoid log of zero
            C_log_values = np.array(all_C_values)
            S_log_values = np.array(all_S_values)
            C_log_values = C_log_values[C_log_values > 0]
            S_log_values = S_log_values[S_log_values > 0]
            if len(C_log_values) > 0 and len(S_log_values) > 0:
                ax4.scatter(np.log10(C_log_values), np.log10(S_log_values), label='All Data Points', color='#66b3ff',
                            alpha=0.5, s=15)

            C_log_values_selected = np.array(C_values)
            S_log_values_selected = np.array(S_values)
            C_log_values_selected = C_log_values_selected[C_log_values_selected > 0]
            S_log_values_selected = S_log_values_selected[S_log_values_selected > 0]
            if len(C_log_values_selected) > 0 and len(S_log_values_selected) > 0:
                # Calculate standard deviations
                unique_c_values = sorted(list(set(C_values)))
                S_log_std = []
                for c_val in unique_c_values:
                    s_log_values_for_c = [np.log10(S_values[i]) for i, c in enumerate(C_values) if
                                          c == c_val and S_values[i] > 0]
                    if len(s_log_values_for_c) > 1:
                        S_log_std.append(np.std(s_log_values_for_c))
                    else:
                        S_log_std.append(0)

                C_log_values_selected_plot = np.array(C_log_values_selected)[np.array(C_log_values_selected) > 0]
                S_log_values_selected_plot = np.array(S_log_values_selected)[np.array(S_log_values_selected) > 0]

                if len(C_log_values_selected_plot) > 0 and len(S_log_values_selected_plot) > 0:
                    ax4.errorbar(np.log10(unique_c_values), [np.mean(
                        [np.log10(S_values[i]) for i, c in enumerate(C_values) if c == c_val and S_values[i] > 0]) for
                                                             c_val in unique_c_values if np.mean(
                            [S_values[i] for i, c in enumerate(C_values) if c == c_val]) > 0], yerr=S_log_std,
                                 fmt='o', label='Selected Data Points', color='#ffa500', markersize=5, capsize=5,
                                 ecolor='black', elinewidth=0.8)

            C_smooth_log = C_smooth[C_smooth > 0]
            S_fit_log = S_fit[C_smooth > 0]
            if len(C_smooth_log) > 0 and len(S_fit_log) > 0:
                ax4.plot(np.log10(C_smooth_log), np.log10(S_fit_log), 'r--', label='Log Scale Fit', linewidth=1.5)

            ax4.set_xlabel('log(Concentration (C)) [unit]', fontsize=10, fontweight='bold')
            ax4.set_ylabel('log(Adsorption Amount (S)) [unit]', fontsize=10, fontweight='bold')
            ax4.set_title('Log Scale Freundlich Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax4.grid(True, linestyle='--', alpha=0.6)
            fig4.patch.set_facecolor('white')

            # Format y-axis ticks with scientific notation if needed
            format_y_axis(ax4)

            # Create a notebook for tabbed plot viewing
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=3, rowspan=10, padx=10, pady=10, sticky='ns')

            # Create frames to hold the canvas widgets
            frame3 = ttk.Frame(notebook)
            frame4 = ttk.Frame(notebook)
            notebook.add(frame3, text='Freundlich Isotherm')
            notebook.add(frame4, text='Freundlich Log Scale')

            # Embed plots into the frames
            canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
            canvas_widget3 = canvas3.get_tk_widget()
            canvas_widget3.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            canvas3.draw()

            canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
            canvas_widget4 = canvas4.get_tk_widget()
            canvas_widget4.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            canvas4.draw()
        else:
            messagebox.showerror("Error", "Please select a valid model (Langmuir or Freundlich).")
    except RuntimeError as e:
        messagebox.showerror("Error", f"Analysis failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


# Refreshes the data display frame
def refresh_data_frame():
    for widget in data_scrollable_frame.winfo_children():  # Use the scrollable frame now
        widget.destroy()

    # Add headers to the dataframe
    tk.Label(data_scrollable_frame, text="No.", font=('Arial', 14, 'bold'), bg='#f0f0f0').grid(row=0, column=1,
                                                                                               padx=(10, 10), pady=2,
                                                                                               sticky='nsew')
    tk.Label(data_scrollable_frame, text="C", font=('Arial', 14, 'bold'), bg='#f0f0f0').grid(row=0, column=2,
                                                                                             padx=(40, 20), pady=2,
                                                                                             sticky='nsew')
    tk.Label(data_scrollable_frame, text="S", font=('Arial', 14, 'bold'), bg='#f0f0f0').grid(row=0, column=3,
                                                                                             padx=(40, 50), pady=2,
                                                                                             sticky='nsew')

    for i, (c, s, group_id) in enumerate(data_list):
        c_str = f"{c:.4f}"  # Format C to 4 decimal places
        s_str = f"{s:.4f}"  # Format S to 4 decimal places
        group_str = f"{group_id}"  # Format group id

        group_label = tk.Label(data_scrollable_frame, text=group_str, font=('Arial', 14),
                               bg='#f0f0f0')  # add group label
        c_label = tk.Label(data_scrollable_frame, text=c_str, font=('Arial', 14), bg='#f0f0f0')
        s_label = tk.Label(data_scrollable_frame, text=s_str, font=('Arial', 14), bg='#f0f0f0')

        group_label.grid(row=i + 1, column=1, padx=(20, 10), pady=2, sticky='e')  # Adjust padx for alignment
        c_label.grid(row=i + 1, column=2, padx=(30, 20), pady=2, sticky='e')  # Adjust padx for alignment
        s_label.grid(row=i + 1, column=3, padx=(30, 50), pady=2, sticky='e')  # Adjust padx for alignment

        tk.Checkbutton(data_scrollable_frame, variable=check_vars[i], bg='#f0f0f0').grid(row=i + 1, column=0, sticky='w'
                                                                                         , padx=(30, 5), pady=2)

    # Update canvas scroll region to fit data_frame size
    data_frame_canvas.update_idletasks()
    data_frame_canvas.config(scrollregion=data_frame_canvas.bbox("all"))


# Function to toggle all checkboxes
def toggle_all():
    all_selected = all(var.get() for var in check_vars.values())
    for var in check_vars.values():
        var.set(not all_selected)


# GUI setup
window = ThemedTk(theme="xpnative")
window.title("Adsorption Analysis")
window.configure(background='white')
style = ttk.Style(window)
style.configure('TButton', font=('Arial', 10), borderwidth=0, relief='flat', background='#4CAF50', foreground='#000000',
                padding=5)  # Button font size reduce
# Style for buttons
button_style = {
    'font': ('Arial', 10),
    'borderwidth': 0,
    'relief': 'flat',
    'background': '#4CAF50',
    'foreground': '#000000',
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
    'borderwidth': 0,  # Removed borderwidth
    'relief': 'solid',
    'highlightthickness': 0,
    'background': 'white'
}
# Layout adjustments
input_frame = ttk.Frame(window)  # Wrap input area in a frame
input_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

# Model selection dropdown
tk.Label(input_frame, text="Select Model : ", **label_style).grid(row=0, column=0, padx=2, pady=2, sticky='e')
model_var = tk.StringVar(window)
model_var.set("Langmuir")  # Default value
model_dropdown = ttk.Combobox(input_frame, textvariable=model_var, values=["Langmuir", "Freundlich"],
                              font=('Arial', 12), width=26)  # apply font here
model_dropdown.grid(row=0, column=1, padx=2, pady=2, sticky='w')

tk.Label(input_frame, text="Concentration (C) : ", **label_style).grid(row=1, column=0, padx=2, pady=2, sticky='e')
c_entry = tk.Entry(input_frame, **entry_style, width=28)
c_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')

tk.Label(input_frame, text="Adsorption (S) : ", **label_style).grid(row=2, column=0, padx=2, pady=2, sticky='e')
s_entry = tk.Entry(input_frame, **entry_style, width=28)
s_entry.grid(row=2, column=1, padx=2, pady=2, sticky='w')

# Create a frame to hold both buttons
button_frame = tk.Frame(input_frame)
button_frame.grid(row=3, column=1, padx=2, pady=2, sticky='e')

add_button = ttk.Button(button_frame, text="Add", command=add_data, style='TButton', width=7)
add_button.grid(row=0, column=0, padx=2, pady=2)

delete_button = ttk.Button(button_frame, text="Delete", command=delete_data, style='TButton', width=7)
delete_button.grid(row=0, column=1, padx=2, pady=2)

select_all_button = ttk.Button(input_frame, text="Select All", command=toggle_all, style='TButton', width=10)
select_all_button.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky='w')

# Scrollable Frame Setup
data_frame_container = tk.Frame(window, bg='white')  # container for both canvas and scrollbar
data_frame_container.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')

data_frame_canvas = tk.Canvas(data_frame_container, bg='white')
data_frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

data_frame_scrollbar = ttk.Scrollbar(data_frame_container, orient=tk.VERTICAL, command=data_frame_canvas.yview)
data_frame_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

data_frame_canvas.configure(yscrollcommand=data_frame_scrollbar.set)
data_frame_canvas.bind("<Configure>", lambda e: data_frame_canvas.configure(scrollregion=data_frame_canvas.bbox("all")))

data_scrollable_frame = tk.Frame(data_frame_canvas, bg='#f0f0f0')  # use this frame instead of the original
data_frame_canvas.create_window((0, 0), window=data_scrollable_frame, anchor="nw")

data_list = []
check_vars = {}

analyze_button = ttk.Button(window, text="Analyze", command=analyze_data, style='TButton', width=14)
analyze_button.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

result_text = tk.Text(window, height=4, width=40, state=tk.DISABLED, font=('Arial', 12), bg='#f0f0f0')
result_text.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

window.grid_columnconfigure(3, weight=1)
window.grid_rowconfigure(5, weight=1)
window.grid_rowconfigure(8, weight=1)
window.grid_columnconfigure(0, weight=0)
window.grid_columnconfigure(3, weight=1)  # Make column 3 (plot area) expand
window.grid_rowconfigure(0, weight=0)  # make the input area fixed height

window.mainloop()