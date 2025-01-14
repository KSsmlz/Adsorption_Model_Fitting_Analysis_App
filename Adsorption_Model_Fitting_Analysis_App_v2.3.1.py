import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain") # change from 'sci' to 'plain' to prevent scientific notation
    # for tick in ax.get_yticklabels():
    #     if len(tick.get_text()) >= 5:
    #         offset = formatter.get_offset()
    #         tick.set_text(f"{tick.get_text()} × {offset}")

# Linear adsorption model equation
def linear_adsorption(C, Ka):
    return Ka * C

# Analyzes adsorption data using the Linear model
def analyze_linear(C, S):
    C = np.array(C, dtype=float)
    S = np.array(S, dtype=float)

    try:
        popt, pcov, *rest = curve_fit(linear_adsorption, C, S, p0=[1.0])
    except RuntimeError:
        raise RuntimeError("Optimal parameters not found: check input values or try to use another model")
    Ka_fit = popt[0]
    fitted_params = [Ka_fit]

    C_smooth = np.linspace(min(C), max(C), 500)
    S_fit = linear_adsorption(C_smooth, Ka_fit)

    S_fit_original = linear_adsorption(C, Ka_fit)
    r_squared = r2_score(S, S_fit_original)

    return fitted_params, r_squared, C_smooth, S_fit


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
        if model_type == "Linear":
            fitted_params, r_squared, C_smooth, S_fit = analyze_linear(C_values, S_values)
            Ka_fit = fitted_params[0]
            result_text.config(state=tk.NORMAL)
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, f"Fitted Linear Equation:\n")
            result_text.insert(tk.END, f"S = {Ka_fit:.3f} * C\n")
            result_text.insert(tk.END, f"R-squared: {r_squared:.4f}")
            result_text.config(state=tk.DISABLED)

            # Create subplots for both regular plot
            fig5, ax5 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI
            ax5.scatter(all_C_values, all_S_values, label='All Data Points', color='#66b3ff', alpha=0.5, s=15)
            ax5.scatter(C_values, S_values, label='Selected Data Points', color='#008000', s=20)

            ax5.plot(C_smooth, S_fit, 'r-', label='Linear Fit', linewidth=1.5)
            ax5.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Adsorption Amount (S) [unit]', fontsize=10, fontweight='bold')
            ax5.set_title('Linear Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax5.grid(True, linestyle='--', alpha=0.6)
            fig5.patch.set_facecolor('white')

            # Format y-axis ticks with scientific notation if needed
            format_y_axis(ax5)

            # 找到 X 和 Y 軸的最大值
            max_C_l = max(all_C_values)
            max_S_l = max(all_S_values)

            # 設定 X 和 Y 軸的邊界，並留一些空間
            ax5.set_xlim(0, max_C_l * 1.1)
            ax5.set_ylim(0, max_S_l * 1.1)

            # Store the initial limits for reset
            initial_xlim5 = ax5.get_xlim()
            initial_ylim5 = ax5.get_ylim()

            # Create a notebook for tabbed plot viewing
            notebook = ttk.Notebook(window)
            notebook.grid(row=0, column=3, rowspan=10, padx=10, pady=10, sticky='ns')

            # Create frames to hold the canvas widgets
            frame5 = ttk.Frame(notebook)
            notebook.add(frame5, text='Linear Isotherm')

            # Embed plots into the frames
            canvas5 = FigureCanvasTkAgg(fig5, master=frame5)
            canvas_widget5 = canvas5.get_tk_widget()
            canvas_widget5.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            canvas5.draw()

            # Navigation toolbars
            toolbar5 = NavigationToolbar2Tk(canvas5, frame5)
            toolbar5.update()
            toolbar5.pack(side=tk.BOTTOM, fill=tk.X)

            # Function to set up the axes ticks
            def set_axis_ticks5(ax, x_ticks, y_ticks):
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_ticks))
                canvas5.draw()

            # Create a spinbox to control x-axis ticks
            x_ticks_var5 = tk.IntVar(value=5)  # Set initial value for x-axis ticks
            x_ticks_label5 = ttk.Label(frame5, text="X Ticks:")
            x_ticks_label5.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            x_ticks_spinbox5 = ttk.Spinbox(frame5, from_=2, to=20, textvariable=x_ticks_var5, width=3,
                                           command=lambda: set_axis_ticks5(ax5, x_ticks_var5.get(), y_ticks_var5.get()))
            x_ticks_spinbox5.pack(side=tk.LEFT, pady=5, padx=2)

            # Create a spinbox to control y-axis ticks
            y_ticks_var5 = tk.IntVar(value=5)  # Set initial value for y-axis ticks
            y_ticks_label5 = ttk.Label(frame5, text="Y Ticks:")
            y_ticks_label5.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            y_ticks_spinbox5 = ttk.Spinbox(frame5, from_=2, to=20, textvariable=y_ticks_var5, width=3,
                                           command=lambda: set_axis_ticks5(ax5, x_ticks_var5.get(), y_ticks_var5.get()))
            y_ticks_spinbox5.pack(side=tk.LEFT, pady=5, padx=2)

            # 新增顯示/隱藏座標按鈕及邏輯
            show_coords5 = False
            annotations5 = []

            def toggle_coords5():
                nonlocal show_coords5
                nonlocal annotations5
                show_coords5 = not show_coords5

                if show_coords5:
                    annotations5 = []
                    for c, s in zip(C_values, S_values):
                        c_str = f"{c:.4f}" if c < 10 else f"{c:.1f}"
                        s_str = f"{s:.4f}" if s < 10 else f"{s:.1f}"
                        annotation = ax5.annotate(f'({c_str}, {s_str})', xy=(c, s), xytext=(3, 3),
                                                  textcoords='offset points', fontsize=8)
                        annotations5.append(annotation)
                    coords_button5.config(text="Hide Coordinates")
                else:
                    for annotation in annotations5:
                        annotation.remove()
                    annotations5 = []
                    coords_button5.config(text="Show Coordinates")
                canvas5.draw()

            # Zoom In function for plot5
            def zoom_in5():
                current_xlim = ax5.get_xlim()
                current_ylim = ax5.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax5.set_xlim(x_center - x_range / 1.2, x_center + x_range / 1.2)
                ax5.set_ylim(y_center - y_range / 1.2, y_center + y_range / 1.2)
                canvas5.draw()

            zoom_in_button5 = ttk.Button(frame5, text="+", command=zoom_in5, style='TButton', width=3)
            zoom_in_button5.pack(side=tk.LEFT, pady=5, padx=2)

            # Zoom Out function for plot5
            def zoom_out5():
                current_xlim = ax5.get_xlim()
                current_ylim = ax5.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax5.set_xlim(x_center - x_range * 1.2, x_center + x_range * 1.2)
                ax5.set_ylim(y_center - y_range * 1.2, y_center + y_range * 1.2)
                canvas5.draw()

            zoom_out_button5 = ttk.Button(frame5, text="-", command=zoom_out5, style='TButton', width=3)
            zoom_out_button5.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view all data points
            def view_all_data5():
                if all_C_values and all_S_values:
                    min_x = min(all_C_values)
                    max_x = max(all_C_values)
                    min_y = min(all_S_values)
                    max_y = max(all_S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax5.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax5.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas5.draw()

            view_all_button5 = ttk.Button(frame5, text="View All", command=view_all_data5, style='TButton', width=10)
            view_all_button5.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view selected data points
            def view_selected_data5():
                if C_values and S_values:
                    min_x = min(C_values)
                    max_x = max(C_values)
                    min_y = min(S_values)
                    max_y = max(S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2
                    ax5.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax5.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas5.draw()

            view_selected_button5 = ttk.Button(frame5, text="View Selected", command=view_selected_data5,
                                               style='TButton', width=12)
            view_selected_button5.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to align axes to data
            def align_axes5():
                current_xlim = ax5.get_xlim()
                current_ylim = ax5.get_ylim()

                # Filter data points within current axes limits
                visible_C = [c for c in all_C_values if current_xlim[0] <= c <= current_xlim[1]]
                visible_S = [s for s in all_S_values if current_ylim[0] <= s <= current_ylim[1]]

                if visible_C:
                    min_x = min(visible_C)
                    max_x = max(visible_C)

                    if min_x >= 0:
                        ax5.set_xlim(0, max_x * 1.1)
                    elif max_x <= 0:
                        ax5.set_xlim(min_x * 1.1, 0)
                    else:
                        ax5.set_xlim(min_x * 1.1, max_x * 1.1)
                if visible_S:
                    min_y = min(visible_S)
                    max_y = max(visible_S)

                    if min_y >= 0:
                        ax5.set_ylim(0, max_y * 1.1)
                    elif max_y <= 0:
                        ax5.set_ylim(min_y * 1.1, 0)
                    else:
                        ax5.set_ylim(min_y * 1.1, max_y * 1.1)
                canvas5.draw()

            align_axes_button5 = ttk.Button(frame5, text="Align Axes", command=align_axes5, style='TButton', width=10)
            align_axes_button5.pack(side=tk.LEFT, pady=5, padx=2)

            coords_button5 = ttk.Button(frame5, text="Show Coordinates", command=toggle_coords5, style='TButton',
                                        width=15)
            coords_button5.pack(side=tk.LEFT, pady=5, padx=2)  # Modified pack here

        elif model_type == "Langmuir":
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
            ax1.scatter(C_values, S_values, label='Selected Data Points', color='#008000', s=20)


            ax1.plot(C_smooth, S_fit, 'r-', label='Langmuir Fit', linewidth=1.5)
            ax1.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Adsorption Amount (S) [unit]', fontsize=10, fontweight='bold')
            ax1.set_title('Langmuir Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax1.grid(True, linestyle='--', alpha=0.6)
            fig1.patch.set_facecolor('white')  # Setting background color for the figure

            format_y_axis(ax1)

            # 找到 X 和 Y 軸的最大值
            max_C = max(all_C_values)
            max_S = max(all_S_values)

            # 設定 X 和 Y 軸的邊界，並留一些空間
            ax1.set_xlim(0, max_C * 1.1)
            ax1.set_ylim(0, max_S * 1.1)

            # Store the initial limits for reset
            initial_xlim1 = ax1.get_xlim()
            initial_ylim1 = ax1.get_ylim()


            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)  # Adjust figure size and DPI
            ax2.scatter(all_C_values, np.array(all_C_values) / np.array(all_S_values), label='All Data Points',
                        color='#66b3ff', alpha=0.5, s=15)
            ax2.scatter(C_values, np.array(C_values) / np.array(S_values), label='Selected Data Points',
                        color='#ffa500', s=20)

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

            # 找到 C 和 C/S 軸的最大值
            max_C_cs = max(all_C_values)
            max_CS = max(np.array(all_C_values) / np.array(all_S_values) if all_S_values else [1])

            # 設定 X 和 Y 軸的邊界，並留一些空間
            ax2.set_xlim(0, max_C_cs * 1.1)
            ax2.set_ylim(0, max_CS * 1.1)

            # Store the initial limits for reset
            initial_xlim2 = ax2.get_xlim()
            initial_ylim2 = ax2.get_ylim()


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

            # Navigation toolbars
            toolbar1 = NavigationToolbar2Tk(canvas1, frame1)
            toolbar1.update()
            toolbar1.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
            toolbar2.update()
            toolbar2.pack(side=tk.BOTTOM, fill=tk.X)

            # Function to set up the axes ticks
            def set_axis_ticks1(ax, x_ticks, y_ticks):
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_ticks))
                canvas1.draw()

            # Create a spinbox to control x-axis ticks
            x_ticks_var1 = tk.IntVar(value=5)  # Set initial value for x-axis ticks
            x_ticks_label1 = ttk.Label(frame1, text="X Ticks:")
            x_ticks_label1.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            x_ticks_spinbox1 = ttk.Spinbox(frame1, from_=2, to=20, textvariable=x_ticks_var1, width=3,
                                           command=lambda: set_axis_ticks1(ax1, x_ticks_var1.get(), y_ticks_var1.get()))
            x_ticks_spinbox1.pack(side=tk.LEFT, pady=5, padx=2)

            # Create a spinbox to control y-axis ticks
            y_ticks_var1 = tk.IntVar(value=5)  # Set initial value for y-axis ticks
            y_ticks_label1 = ttk.Label(frame1, text="Y Ticks:")
            y_ticks_label1.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            y_ticks_spinbox1 = ttk.Spinbox(frame1, from_=2, to=20, textvariable=y_ticks_var1, width=3,
                                           command=lambda: set_axis_ticks1(ax1, x_ticks_var1.get(), y_ticks_var1.get()))
            y_ticks_spinbox1.pack(side=tk.LEFT, pady=5, padx=2)

            # 新增顯示/隱藏座標按鈕及邏輯
            show_coords1 = False
            annotations1 = []

            def toggle_coords1():
                nonlocal show_coords1
                nonlocal annotations1
                show_coords1 = not show_coords1

                if show_coords1:
                    annotations1 = []
                    for c, s in zip(C_values, S_values):
                        c_str = f"{c:.4f}" if c < 10 else f"{c:.1f}"
                        s_str = f"{s:.4f}" if s < 10 else f"{s:.1f}"
                        annotation = ax1.annotate(f'({c_str}, {s_str})', xy=(c, s), xytext=(3, 3),
                                                  textcoords='offset points', fontsize=8)
                        annotations1.append(annotation)
                    coords_button1.config(text="Hide Coordinates")
                else:
                    for annotation in annotations1:
                        annotation.remove()
                    annotations1 = []
                    coords_button1.config(text="Show Coordinates")
                canvas1.draw()

            # Zoom In function for plot1
            def zoom_in1():
                current_xlim = ax1.get_xlim()
                current_ylim = ax1.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax1.set_xlim(x_center - x_range/1.2, x_center + x_range/1.2)
                ax1.set_ylim(y_center - y_range/1.2, y_center + y_range/1.2)
                canvas1.draw()

            zoom_in_button1 = ttk.Button(frame1, text="+", command=zoom_in1, style='TButton', width=3)
            zoom_in_button1.pack(side=tk.LEFT, pady=5, padx=2)

            # Zoom Out function for plot1
            def zoom_out1():
                current_xlim = ax1.get_xlim()
                current_ylim = ax1.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax1.set_xlim(x_center - x_range*1.2, x_center + x_range*1.2)
                ax1.set_ylim(y_center - y_range*1.2, y_center + y_range*1.2)
                canvas1.draw()

            zoom_out_button1 = ttk.Button(frame1, text="-", command=zoom_out1, style='TButton', width=3)
            zoom_out_button1.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view all data points
            def view_all_data1():
                if all_C_values and all_S_values:
                    min_x = min(all_C_values)
                    max_x = max(all_C_values)
                    min_y = min(all_S_values)
                    max_y = max(all_S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax1.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax1.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas1.draw()

            view_all_button1 = ttk.Button(frame1, text="View All", command=view_all_data1, style='TButton', width=10)
            view_all_button1.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view selected data points
            def view_selected_data1():
                if C_values and S_values:
                    min_x = min(C_values)
                    max_x = max(C_values)
                    min_y = min(S_values)
                    max_y = max(S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2
                    ax1.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax1.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas1.draw()

            view_selected_button1 = ttk.Button(frame1, text="View Selected", command=view_selected_data1,
                                               style='TButton', width=12)
            view_selected_button1.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to align axes to data
            def align_axes1():
                current_xlim = ax1.get_xlim()
                current_ylim = ax1.get_ylim()

                # Filter data points within current axes limits
                visible_C = [c for c in all_C_values if current_xlim[0] <= c <= current_xlim[1]]
                visible_S = [s for s in all_S_values if current_ylim[0] <= s <= current_ylim[1]]

                if visible_C:
                    min_x = min(visible_C)
                    max_x = max(visible_C)

                    if min_x >= 0:
                        ax1.set_xlim(0, max_x * 1.1)
                    elif max_x <= 0:
                        ax1.set_xlim(min_x * 1.1, 0)
                    else:
                        ax1.set_xlim(min_x * 1.1, max_x * 1.1)
                if visible_S:
                    min_y = min(visible_S)
                    max_y = max(visible_S)

                    if min_y >= 0:
                        ax1.set_ylim(0, max_y * 1.1)
                    elif max_y <= 0:
                        ax1.set_ylim(min_y * 1.1, 0)
                    else:
                        ax1.set_ylim(min_y * 1.1, max_y * 1.1)

                canvas1.draw()

            align_axes_button1 = ttk.Button(frame1, text="Align Axes", command=align_axes1, style='TButton', width=10)
            align_axes_button1.pack(side=tk.LEFT, pady=5, padx=2)

            coords_button1 = ttk.Button(frame1, text="Show Coordinates", command=toggle_coords1, style='TButton',
                                        width=15)
            coords_button1.pack(side=tk.LEFT, pady=5, padx=2)  # Modified pack here

            # Function to set up the axes ticks
            def set_axis_ticks2(ax, x_ticks, y_ticks):
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_ticks))
                canvas2.draw()

            # Create a spinbox to control x-axis ticks
            x_ticks_var2 = tk.IntVar(value=5)  # Set initial value for x-axis ticks
            x_ticks_label2 = ttk.Label(frame2, text="X Ticks:")
            x_ticks_label2.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            x_ticks_spinbox2 = ttk.Spinbox(frame2, from_=2, to=20, textvariable=x_ticks_var2, width=3,
                                           command=lambda: set_axis_ticks2(ax2, x_ticks_var2.get(), y_ticks_var2.get()))
            x_ticks_spinbox2.pack(side=tk.LEFT, pady=5, padx=2)

            # Create a spinbox to control y-axis ticks
            y_ticks_var2 = tk.IntVar(value=5)  # Set initial value for y-axis ticks
            y_ticks_label2 = ttk.Label(frame2, text="Y Ticks:")
            y_ticks_label2.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            y_ticks_spinbox2 = ttk.Spinbox(frame2, from_=2, to=20, textvariable=y_ticks_var2, width=3,
                                           command=lambda: set_axis_ticks2(ax2, x_ticks_var2.get(), y_ticks_var2.get()))
            y_ticks_spinbox2.pack(side=tk.LEFT, pady=5, padx=2)

            show_coords2 = False
            annotations2 = []

            def toggle_coords2():
                nonlocal show_coords2
                nonlocal annotations2
                show_coords2 = not show_coords2

                if show_coords2:
                    annotations2 = []
                    for c, cs in zip(C_values, np.array(C_values) / np.array(S_values)):
                        c_str = f"{c:.4f}" if c < 10 else f"{c:.1f}"
                        cs_str = f"{cs:.4f}" if cs < 10 else f"{cs:.1f}"
                        annotation = ax2.annotate(f'({c_str}, {cs_str})', xy=(c, cs), xytext=(3, 3),
                                                  textcoords='offset points', fontsize=8)
                        annotations2.append(annotation)
                    coords_button2.config(text="Hide Coordinates")
                else:
                    for annotation in annotations2:
                        annotation.remove()
                    annotations2 = []
                    coords_button2.config(text="Show Coordinates")
                canvas2.draw()


            # Zoom In function for plot2
            def zoom_in2():
                current_xlim = ax2.get_xlim()
                current_ylim = ax2.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax2.set_xlim(x_center - x_range/1.2, x_center + x_range/1.2)
                ax2.set_ylim(y_center - y_range/1.2, y_center + y_range/1.2)
                canvas2.draw()

            zoom_in_button2 = ttk.Button(frame2, text="+", command=zoom_in2, style='TButton', width=3)
            zoom_in_button2.pack(side=tk.LEFT, pady=5, padx=2)

             # Zoom Out function for plot2
            def zoom_out2():
                current_xlim = ax2.get_xlim()
                current_ylim = ax2.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax2.set_xlim(x_center - x_range*1.2, x_center + x_range*1.2)
                ax2.set_ylim(y_center - y_range*1.2, y_center + y_range*1.2)
                canvas2.draw()

            zoom_out_button2 = ttk.Button(frame2, text="-", command=zoom_out2, style='TButton', width=3)
            zoom_out_button2.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view all data points
            def view_all_data2():
                 if all_C_values and all_S_values:
                    cs_values = np.array(all_C_values) / np.array(all_S_values) if all_S_values else []

                    if cs_values.size > 0:
                       min_x = min(all_C_values)
                       max_x = max(all_C_values)
                       min_y = min(cs_values)
                       max_y = max(cs_values)

                       x_range = max_x - min_x
                       y_range = max_y - min_y
                       x_center = (max_x + min_x)/2
                       y_center = (max_y + min_y)/2

                       ax2.set_xlim(x_center - x_range*0.6, x_center + x_range*0.6)
                       ax2.set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
                       canvas2.draw()

            view_all_button2 = ttk.Button(frame2, text="View All", command=view_all_data2, style='TButton', width=10)
            view_all_button2.pack(side=tk.LEFT, pady=5, padx=2)

             # Function to view selected data points
            def view_selected_data2():
                 if C_values and S_values:
                    cs_values = np.array(C_values) / np.array(S_values)
                    if cs_values.size > 0:
                       min_x = min(C_values)
                       max_x = max(C_values)
                       min_y = min(cs_values)
                       max_y = max(cs_values)

                       x_range = max_x - min_x
                       y_range = max_y - min_y
                       x_center = (max_x + min_x)/2
                       y_center = (max_y + min_y)/2
                       ax2.set_xlim(x_center - x_range*0.6, x_center + x_range*0.6)
                       ax2.set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
                       canvas2.draw()

            view_selected_button2 = ttk.Button(frame2, text="View Selected", command=view_selected_data2, style='TButton', width=12)
            view_selected_button2.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to align axes to data
            def align_axes2():
                current_xlim = ax2.get_xlim()
                current_ylim = ax2.get_ylim()

                cs_values = np.array(all_C_values) / np.array(all_S_values) if all_S_values else []
                if cs_values.size > 0:
                    # Filter data points within current axes limits
                    visible_C = [c for c, cs in zip(all_C_values, cs_values) if
                                 current_xlim[0] <= c <= current_xlim[1] and current_ylim[0] <= cs <= current_ylim[1]]
                    visible_CS = [cs for c, cs in zip(all_C_values, cs_values) if
                                  current_xlim[0] <= c <= current_xlim[1] and current_ylim[0] <= cs <= current_ylim[1]]

                    if visible_C:
                        min_x = min(visible_C)
                        max_x = max(visible_C)

                        if min_x >= 0:
                            ax2.set_xlim(0, max_x * 1.1)
                        elif max_x <= 0:
                            ax2.set_xlim(min_x * 1.1, 0)
                        else:
                            ax2.set_xlim(min_x * 1.1, max_x * 1.1)

                    if visible_CS:
                        min_y = min(visible_CS)
                        max_y = max(visible_CS)

                        if min_y >= 0:
                            ax2.set_ylim(0, max_y * 1.1)
                        elif max_y <= 0:
                            ax2.set_ylim(min_y * 1.1, 0)
                        else:
                            ax2.set_ylim(min_y * 1.1, max_y * 1.1)
                    canvas2.draw()

            align_axes_button2 = ttk.Button(frame2, text="Align Axes", command=align_axes2, style='TButton', width=10)
            align_axes_button2.pack(side=tk.LEFT, pady=5, padx=2)

            coords_button2 = ttk.Button(frame2, text="Show Coordinates", command=toggle_coords2, style='TButton',
                                        width=15)
            coords_button2.pack(side=tk.LEFT, pady=5, padx=2)  # Modified pack here


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
            ax3.scatter(C_values, S_values, label='Selected Data Points', color='#008000', s=20)

            ax3.plot(C_smooth, S_fit, 'r-', label='Freundlich Fit', linewidth=1.5)
            ax3.set_xlabel('Concentration (C) [unit]', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Adsorption Amount (S) [unit]', fontsize=10, fontweight='bold')
            ax3.set_title('Freundlich Adsorption Isotherm', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='grey', framealpha=0.8)
            ax3.grid(True, linestyle='--', alpha=0.6)
            fig3.patch.set_facecolor('white')

            # Format y-axis ticks with scientific notation if needed
            format_y_axis(ax3)

            # 找到 X 和 Y 軸的最大值
            max_C_f = max(all_C_values)
            max_S_f = max(all_S_values)

            # 設定 X 和 Y 軸的邊界，並留一些空間
            ax3.set_xlim(0, max_C_f * 1.1)
            ax3.set_ylim(0, max_S_f * 1.1)

            # Store the initial limits for reset
            initial_xlim3 = ax3.get_xlim()
            initial_ylim3 = ax3.get_ylim()

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
                ax4.scatter(np.log10(C_log_values_selected), np.log10(S_log_values_selected),
                            label='Selected Data Points', color='#ffa500', s=20)

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

            # Store the initial limits for reset
            initial_xlim4 = ax4.get_xlim()
            initial_ylim4 = ax4.get_ylim()

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


            # Navigation toolbars
            toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
            toolbar3.update()
            toolbar3.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar4 = NavigationToolbar2Tk(canvas4, frame4)
            toolbar4.update()
            toolbar4.pack(side=tk.BOTTOM, fill=tk.X)

            # Function to set up the axes ticks
            def set_axis_ticks3(ax, x_ticks, y_ticks):
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_ticks))
                canvas3.draw()


            # Create a spinbox to control x-axis ticks
            x_ticks_var3 = tk.IntVar(value=5)  # Set initial value for x-axis ticks
            x_ticks_label3 = ttk.Label(frame3, text="X Ticks:")
            x_ticks_label3.pack(side=tk.LEFT, pady=5, padx=(2,0))
            x_ticks_spinbox3 = ttk.Spinbox(frame3, from_=2, to=20, textvariable=x_ticks_var3, width=3,
                                         command=lambda: set_axis_ticks3(ax3, x_ticks_var3.get(), y_ticks_var3.get()))
            x_ticks_spinbox3.pack(side=tk.LEFT, pady=5, padx=2)

            # Create a spinbox to control y-axis ticks
            y_ticks_var3 = tk.IntVar(value=5)  # Set initial value for y-axis ticks
            y_ticks_label3 = ttk.Label(frame3, text="Y Ticks:")
            y_ticks_label3.pack(side=tk.LEFT, pady=5, padx=(2,0))
            y_ticks_spinbox3 = ttk.Spinbox(frame3, from_=2, to=20, textvariable=y_ticks_var3, width=3,
                                         command=lambda: set_axis_ticks3(ax3, x_ticks_var3.get(), y_ticks_var3.get()))
            y_ticks_spinbox3.pack(side=tk.LEFT, pady=5, padx=2)

            # 新增顯示/隱藏座標按鈕及邏輯
            show_coords3 = False
            annotations3 = []

            # 新增顯示/隱藏座標按鈕及邏輯
            show_coords3 = False
            annotations3 = []

            def toggle_coords3():
                nonlocal show_coords3
                nonlocal annotations3
                show_coords3 = not show_coords3

                if show_coords3:
                    annotations3 = []
                    for c, s in zip(C_values, S_values):
                        c_str = f"{c:.4f}" if c < 10 else f"{c:.1f}"
                        s_str = f"{s:.4f}" if s < 10 else f"{s:.1f}"
                        annotation = ax3.annotate(f'({c_str}, {s_str})', xy=(c, s), xytext=(3, 3),
                                                  textcoords='offset points', fontsize=8)
                        annotations3.append(annotation)
                    coords_button3.config(text="Hide Coordinates")
                else:
                    for annotation in annotations3:
                        annotation.remove()
                    annotations3 = []
                    coords_button3.config(text="Show Coordinates")
                canvas3.draw()


            # Zoom In function for plot3
            def zoom_in3():
                current_xlim = ax3.get_xlim()
                current_ylim = ax3.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax3.set_xlim(x_center - x_range / 1.2, x_center + x_range / 1.2)
                ax3.set_ylim(y_center - y_range / 1.2, y_center + y_range / 1.2)
                canvas3.draw()

            zoom_in_button3 = ttk.Button(frame3, text="+", command=zoom_in3, style='TButton', width=3)
            zoom_in_button3.pack(side=tk.LEFT, pady=5, padx=2)

            # Zoom Out function for plot3
            def zoom_out3():
                current_xlim = ax3.get_xlim()
                current_ylim = ax3.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax3.set_xlim(x_center - x_range * 1.2, x_center + x_range * 1.2)
                ax3.set_ylim(y_center - y_range * 1.2, y_center + y_range * 1.2)
                canvas3.draw()

            zoom_out_button3 = ttk.Button(frame3, text="-", command=zoom_out3, style='TButton', width=3)
            zoom_out_button3.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view all data points
            def view_all_data3():
                if all_C_values and all_S_values:
                    min_x = min(all_C_values)
                    max_x = max(all_C_values)
                    min_y = min(all_S_values)
                    max_y = max(all_S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax3.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax3.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas3.draw()

            view_all_button3 = ttk.Button(frame3, text="View All", command=view_all_data3, style='TButton', width=10)
            view_all_button3.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view selected data points
            def view_selected_data3():
                if C_values and S_values:
                    min_x = min(C_values)
                    max_x = max(C_values)
                    min_y = min(S_values)
                    max_y = max(S_values)

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax3.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax3.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas3.draw()

            view_selected_button3 = ttk.Button(frame3, text="View Selected", command=view_selected_data3,
                                               style='TButton', width=12)
            view_selected_button3.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to align axes to data
            def align_axes3():
                current_xlim = ax3.get_xlim()
                current_ylim = ax3.get_ylim()

                # Filter data points within current axes limits
                visible_C = [c for c in all_C_values if current_xlim[0] <= c <= current_xlim[1]]
                visible_S = [s for s in all_S_values if current_ylim[0] <= s <= current_ylim[1]]

                if visible_C:
                    min_x = min(visible_C)
                    max_x = max(visible_C)

                    if min_x >= 0:
                        ax3.set_xlim(0, max_x * 1.1)
                    elif max_x <= 0:
                        ax3.set_xlim(min_x * 1.1, 0)
                    else:
                        ax3.set_xlim(min_x * 1.1, max_x * 1.1)

                if visible_S:
                    min_y = min(visible_S)
                    max_y = max(visible_S)

                    if min_y >= 0:
                        ax3.set_ylim(0, max_y * 1.1)
                    elif max_y <= 0:
                        ax3.set_ylim(min_y * 1.1, 0)
                    else:
                        ax3.set_ylim(min_y * 1.1, max_y * 1.1)
                canvas3.draw()

            align_axes_button3 = ttk.Button(frame3, text="Align Axes", command=align_axes3, style='TButton', width=10)
            align_axes_button3.pack(side=tk.LEFT, pady=5, padx=2)

            coords_button3 = ttk.Button(frame3, text="Show Coordinates", command=toggle_coords3, style='TButton',
                                        width=15)
            coords_button3.pack(side=tk.LEFT, pady=5, padx=2)  # Modified pack here

            # Function to set up the axes ticks
            def set_axis_ticks4(ax, x_ticks, y_ticks):
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_ticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_ticks))
                canvas4.draw()


            # Create a spinbox to control x-axis ticks
            x_ticks_var4 = tk.IntVar(value=5)  # Set initial value for x-axis ticks
            x_ticks_label4 = ttk.Label(frame4, text="X Ticks:")
            x_ticks_label4.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            x_ticks_spinbox4 = ttk.Spinbox(frame4, from_=2, to=20, textvariable=x_ticks_var4, width=3,
                                         command=lambda: set_axis_ticks4(ax4, x_ticks_var4.get(), y_ticks_var4.get()))
            x_ticks_spinbox4.pack(side=tk.LEFT, pady=5, padx=2)

            # Create a spinbox to control y-axis ticks
            y_ticks_var4 = tk.IntVar(value=5)  # Set initial value for y-axis ticks
            y_ticks_label4 = ttk.Label(frame4, text="Y Ticks:")
            y_ticks_label4.pack(side=tk.LEFT, pady=5, padx=(2, 0))
            y_ticks_spinbox4 = ttk.Spinbox(frame4, from_=2, to=20, textvariable=y_ticks_var4, width=3,
                                         command=lambda: set_axis_ticks4(ax4, x_ticks_var4.get(), y_ticks_var4.get()))
            y_ticks_spinbox4.pack(side=tk.LEFT, pady=5, padx=2)


            show_coords4 = False
            annotations4 = []

            show_coords4 = False
            annotations4 = []

            def toggle_coords4():
                nonlocal show_coords4
                nonlocal annotations4
                show_coords4 = not show_coords4

                if show_coords4:
                    annotations4 = []
                    for c, s in zip(C_log_values_selected, S_log_values_selected):
                        c_str = f"{c:.4f}" if c < 10 else f"{c:.1f}"
                        s_str = f"{s:.4f}" if s < 10 else f"{s:.1f}"
                        annotation = ax4.annotate(f'({c_str}, {s_str})', xy=(np.log10(c), np.log10(s)), xytext=(3, 3),
                                                  textcoords='offset points', fontsize=8)
                        annotations4.append(annotation)
                    coords_button4.config(text="Hide Coordinates")
                else:
                    for annotation in annotations4:
                        annotation.remove()
                    annotations4 = []
                    coords_button4.config(text="Show Coordinates")
                canvas4.draw()


            # Zoom In function for plot4
            def zoom_in4():
                current_xlim = ax4.get_xlim()
                current_ylim = ax4.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax4.set_xlim(x_center - x_range / 1.2, x_center + x_range / 1.2)
                ax4.set_ylim(y_center - y_range / 1.2, y_center + y_range / 1.2)
                canvas4.draw()

            zoom_in_button4 = ttk.Button(frame4, text="+", command=zoom_in4, style='TButton', width=3)
            zoom_in_button4.pack(side=tk.LEFT, pady=5, padx=2)

            # Zoom Out function for plot4
            def zoom_out4():
                current_xlim = ax4.get_xlim()
                current_ylim = ax4.get_ylim()
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (current_ylim[0] + current_ylim[1]) / 2
                x_range = (current_xlim[1] - current_xlim[0]) / 2
                y_range = (current_ylim[1] - current_ylim[0]) / 2
                ax4.set_xlim(x_center - x_range * 1.2, x_center + x_range * 1.2)
                ax4.set_ylim(y_center - y_range * 1.2, y_center + y_range * 1.2)
                canvas4.draw()

            zoom_out_button4 = ttk.Button(frame4, text="-", command=zoom_out4, style='TButton', width=3)
            zoom_out_button4.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view all data points
            def view_all_data4():
                C_log_values = np.array(all_C_values)
                S_log_values = np.array(all_S_values)
                C_log_values = C_log_values[C_log_values > 0]
                S_log_values = S_log_values[S_log_values > 0]

                if len(C_log_values) > 0 and len(S_log_values) > 0:
                    min_x = min(np.log10(C_log_values))
                    max_x = max(np.log10(C_log_values))
                    min_y = min(np.log10(S_log_values))
                    max_y = max(np.log10(S_log_values))

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax4.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax4.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas4.draw()

            view_all_button4 = ttk.Button(frame4, text="View All", command=view_all_data4, style='TButton', width=10)
            view_all_button4.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to view selected data points
            def view_selected_data4():
                C_log_values_selected = np.array(C_values)
                S_log_values_selected = np.array(S_values)
                C_log_values_selected = C_log_values_selected[C_log_values_selected > 0]
                S_log_values_selected = S_log_values_selected[S_log_values_selected > 0]

                if len(C_log_values_selected) > 0 and len(S_log_values_selected) > 0:
                    min_x = min(np.log10(C_log_values_selected))
                    max_x = max(np.log10(C_log_values_selected))
                    min_y = min(np.log10(S_log_values_selected))
                    max_y = max(np.log10(S_log_values_selected))

                    x_range = max_x - min_x
                    y_range = max_y - min_y
                    x_center = (max_x + min_x) / 2
                    y_center = (max_y + min_y) / 2

                    ax4.set_xlim(x_center - x_range * 0.6, x_center + x_range * 0.6)
                    ax4.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
                    canvas4.draw()

            view_selected_button4 = ttk.Button(frame4, text="View Selected", command=view_selected_data4,
                                               style='TButton', width=12)
            view_selected_button4.pack(side=tk.LEFT, pady=5, padx=2)

            # Function to align axes to data
            def align_axes4():
                current_xlim = ax4.get_xlim()
                current_ylim = ax4.get_ylim()

                C_log_values = np.array(all_C_values)
                S_log_values = np.array(all_S_values)
                C_log_values = C_log_values[C_log_values > 0]
                S_log_values = S_log_values[S_log_values > 0]

                if len(C_log_values) > 0 and len(S_log_values) > 0:
                    # Filter data points within current axes limits
                    visible_C = [c for c in C_log_values if current_xlim[0] <= np.log10(c) <= current_xlim[1]]
                    visible_S = [s for s in S_log_values if current_ylim[0] <= np.log10(s) <= current_ylim[1]]
                    if visible_C:
                        min_x = min(np.log10(visible_C))
                        max_x = max(np.log10(visible_C))

                        if min_x >= 0:
                            ax4.set_xlim(0, max_x * 1.1)
                        elif max_x <= 0:
                            ax4.set_xlim(min_x * 1.1, 0)
                        else:
                            ax4.set_xlim(min_x * 1.1, max_x * 1.1)
                    if visible_S:
                        min_y = min(np.log10(visible_S))
                        max_y = max(np.log10(visible_S))

                        if min_y >= 0:
                            ax4.set_ylim(0, max_y * 1.1)
                        elif max_y <= 0:
                            ax4.set_ylim(min_y * 1.1, 0)
                        else:
                            ax4.set_ylim(min_y * 1.1, max_y * 1.1)
                    canvas4.draw()

            align_axes_button4 = ttk.Button(frame4, text="Align Axes", command=align_axes4, style='TButton', width=10)
            align_axes_button4.pack(side=tk.LEFT, pady=5, padx=2)

            coords_button4 = ttk.Button(frame4, text="Show Coordinates", command=toggle_coords4, style='TButton',
                                        width=15)
            coords_button4.pack(side=tk.LEFT, pady=5, padx=2)  # Modified pack here


        else:
            messagebox.showerror("Error", "Please select a valid model (Linear, Langmuir or Freundlich).")
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
model_var.set("Linear")  # Default value
model_dropdown = ttk.Combobox(input_frame, textvariable=model_var, values=["Linear", "Langmuir", "Freundlich"],
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