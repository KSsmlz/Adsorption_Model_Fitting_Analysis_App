import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# Input 5 sets of data for S and C
S = np.array([71.4831, 197.1765, 564.8591, 1824.1754, 677.5149])
C = np.array([0.0062, 0.0195, 0.0427, 0.1888, 1.6131])

# Only use the first 4 sets of data for fitting
S_fit = S[:4]
C_fit = C[:4]


# Define the Langmuir isotherm function
def langmuir_eq(c, KL, b):
    return (KL * b * c) / (1 + KL * c)


# Initial guess for the parameters [KL, b]
initial_guess = [2, 6000]

# Perform non-linear fitting
params_fitted, *rest = curve_fit(langmuir_eq, C_fit, S_fit, p0=initial_guess)

KL_fitted, b_fitted = params_fitted  # Fitted parameters

# Generate the trendline based on the fitted parameters
C_range = np.linspace(min(C_fit), max(C_fit), 100)
S_trend = langmuir_eq(C_range, KL_fitted, b_fitted)

# Calculate R-squared value
S_pred = langmuir_eq(C_fit, KL_fitted, b_fitted)  # Predicted S values
SS_res = np.sum((S_fit - S_pred) ** 2)  # Residual sum of squares
SS_tot = np.sum((S_fit - np.mean(S_fit)) ** 2)  # Total sum of squares
R_squared = 1 - (SS_res / SS_tot)  # R^2 value

# Create Plotly figure
fig = go.Figure()

# Add scatter points for data
fig.add_trace(go.Scatter(x=C, y=S, mode='markers', name='Data points',
                         marker=dict(size=8, color='red')))

# Add fitted trendline
fig.add_trace(go.Scatter(x=C_range, y=S_trend, mode='lines', name='Fitted Trendline',
                         line=dict(width=2, color='blue')))

# Update layout
fig.update_layout(
    title='Langmuir Isotherm Fit',
    xaxis=dict(title='C (mol/L)', x),
    yaxis=dict(title='S (mol/g)', range=[0, max(S)*1.1]),
    showlegend=True
)

# Annotate the equation and R^2 value
formula_text = f"S = ({KL_fitted:.4f} * {b_fitted:.4f} * C) / (1 + {KL_fitted:.4f} * C)"
r2_text = f"R² = {R_squared:.4f}"
fig.add_annotation(x=0.5, y=0.6, text=formula_text, showarrow=False, font=dict(size=12, color="blue"), xref="paper",
                   yref="paper")
fig.add_annotation(x=0.5, y=0.5, text=r2_text, showarrow=False, font=dict(size=12, color="blue"), xref="paper",
                   yref="paper")

# Show plot
fig.show()

# Display the fitted parameters and R^2 value
print('Fitted parameters:')
print(f'KL = {KL_fitted:.4f} ml/mol')
print(f'b = {b_fitted:.4f} mol/g')
print(f'R^2 = {R_squared:.4f}')
