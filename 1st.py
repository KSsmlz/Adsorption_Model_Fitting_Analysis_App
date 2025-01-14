import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定義Langmuir等溫吸附函數
def langmuir_isotherm(C, KL, b):
    return (KL * b * C) / (1 + KL * C)

# 提示使用者輸入4組數據
C = np.array([0.0062, 0.0195, 0.0427, 0.1888])
S = np.array([71.4831, 197.1765, 564.8591, 1824.1754])

# 使用curve_fit進行非線性擬合
initial_guess = [10, 0.5]
params_fitted, _ = curve_fit(langmuir_isotherm, C, S, p0=initial_guess)  # 解包兩個值

KL_fitted = params_fitted[0]  # Langmuir常數 (KL)
b_fitted = params_fitted[1]   # 最大吸附量 (b)

# 根據擬合參數生成趨勢線
C_range = np.linspace(min(C), max(C), 100)
S_trend = langmuir_isotherm(C_range, KL_fitted, b_fitted)

# 計算R²值
S_pred = langmuir_isotherm(C, KL_fitted, b_fitted)  # 預測的S值
SS_res = np.sum((S - S_pred) ** 2)  # 殘差平方和
SS_tot = np.sum((S - np.mean(S)) ** 2)  # 總平方和
R_squared = 1 - (SS_res / SS_tot)  # R²值

# 繪製數據點及趨勢線
plt.figure()
plt.plot(C, S, 'ro', markersize=8, label='數據點')  # 所有數據點
plt.plot(C_range, S_trend, 'b-', linewidth=2, label='擬合趨勢線')
plt.xlabel('C (mol/L)')
plt.ylabel('S (mol/g)')
plt.legend()
plt.title('Langmuir 等溫吸附擬合')
plt.grid(True)

# 顯示擬合參數和R²值
print('擬合參數:')
print(f'KL = {KL_fitted:.4f} ml/mol')
print(f'b = {b_fitted:.4f} mol/g')
print(f'R² = {R_squared:.4f}')

# 在圖上註明方程式和R²值
formula_text = f'S = ({KL_fitted:.4f} * {b_fitted:.4f} * C) / (1 + {KL_fitted:.4f} * C)'
r2_text = f'R² = {R_squared:.4f}'
plt.text(np.mean(C_range) * 0.5, max(S_trend) * 0.7, formula_text, fontsize=12, color='blue', backgroundcolor='white')
plt.text(np.mean(C_range) * 0.5, max(S_trend) * 0.6, r2_text, fontsize=12, color='blue', backgroundcolor='white')

# 繪製C對C/S圖
plt.figure()
plt.plot(C, C/S, 'ro', markersize=8, label='數據點')
plt.xlabel('C (mol/L)')
plt.ylabel('C/S (L/g)')
plt.title('C對C/S圖')
plt.grid(True)

# C對C/S的線性擬合
def linear_fit(C, intercept, slope):
    return intercept + slope * C

intercept, slope = np.polyfit(C, C/S, 1)
C_fit_range = np.linspace(min(C), max(C), 100)
C_over_S_trend = linear_fit(C_fit_range, intercept, slope)
plt.plot(C_fit_range, C_over_S_trend, 'b-', linewidth=2, label='線性擬合趨勢線')
plt.legend()

# 顯示擬合參數和R²值
print('擬合參數 (C對C/S):')
print(f'截距 = {intercept:.4f} L/g')
print(f'斜率 = {slope:.4f} L/mol')

# 顯示圖表
plt.show()
