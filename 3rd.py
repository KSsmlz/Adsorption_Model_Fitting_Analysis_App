import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定義 Langmuir 等溫線公式
def langmuir_isotherm(C, KL, b):
    return (KL * b * C) / (1 + KL * C)

# 主程式
def main():
    # 提示使用者輸入 C 和 S 數據
    print("請輸入吸附實驗的數據：")
    C = list(map(float, input("請輸入液相核種濃度 C（以空格分隔，單位 mol/L）：\n").split()))
    S = list(map(float, input("請輸入固相吸附量 S（以空格分隔，單位 mol/g）：\n").split()))

    # 將輸入的數據轉為 NumPy 陣列
    C = np.array(C)
    S = np.array(S)

    # 檢查數據是否有效
    if len(C) != len(S):
        raise ValueError("C 和 S 數據長度不匹配！請檢查輸入。")
    if any(C <= 0) or any(S <= 0):
        raise ValueError("C 和 S 的數據值必須為正！請檢查輸入。")

    # 初始猜測參數 [KL, b]
    initial_guess = [0.1, 0.1]
    bounds = (0, [np.inf, np.inf])

    # 非線性曲線擬合
    try:
        popt, pcov = curve_fit(langmuir_isotherm, C, S, p0=initial_guess, bounds=bounds)
    except RuntimeError as e:
        print("擬合失敗！請檢查數據是否適合 Langmuir 等溫線模型。")
        print(f"錯誤訊息：{e}")
        return

    # 獲取擬合參數
    KL_fitted, b_fitted = popt

    # 生成擬合曲線
    C_range = np.linspace(min(C), max(C), 100)
    S_fitted = langmuir_isotherm(C_range, KL_fitted, b_fitted)

    # 計算 R^2
    S_pred = langmuir_isotherm(C, KL_fitted, b_fitted)
    SS_res = np.sum((S - S_pred) ** 2)
    SS_tot = np.sum((S - np.mean(S)) ** 2)
    R_squared = 1 - (SS_res / SS_tot)

    # 繪製 C-S 圖與擬合曲線
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(C, S, color='red', label='數據點', zorder=5)
    plt.plot(C_range, S_fitted, color='blue', label='擬合曲線', zorder=3)
    plt.xlabel('C (mol/L)')
    plt.ylabel('S (mol/g)')
    plt.title('Langmuir 等溫線擬合')
    plt.legend()
    plt.grid()

    # 繪製 C-C/S 圖
    C_over_S = C / S
    plt.subplot(1, 2, 2)
    plt.scatter(C, C_over_S, color='green', label='數據點', zorder=5)
    plt.xlabel('C (mol/L)')
    plt.ylabel('C/S')
    plt.title('C-C/S 圖')
    plt.grid()

    # 顯示參數與方程式
    formula_text = f'S = ({KL_fitted:.4f} * {b_fitted:.4f} * C) / (1 + {KL_fitted:.4f} * C)'
    r2_text = f'R^2 = {R_squared:.4f}'
    plt.figtext(0.5, 0.01, formula_text + '\n' + r2_text, ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

    # 顯示圖表
    plt.tight_layout()
    plt.show()

    # 印出結果
    print("擬合結果：")
    print(f"KL = {KL_fitted:.4f} ml/mol")
    print(f"b = {b_fitted:.4f} mol/g")
    print(f"R^2 = {R_squared:.4f}")

# 執行主程式
if __name__ == "__main__":
    main()
