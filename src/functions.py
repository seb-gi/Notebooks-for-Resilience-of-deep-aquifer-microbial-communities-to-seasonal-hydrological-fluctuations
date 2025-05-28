import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =====================================
# === General Utilities ===
# =====================================

def filter_nan(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


# =====================================
# === Linear Fit & Plotting Utilities ===
# =====================================

def plot_linear_fit(ax, x, y, color, label, x_text, y_text):
    x_clean, y_clean = filter_nan(x, y)
    if len(x_clean) > 1:
        b, a = np.polyfit(x_clean, y_clean, deg=1)
        xseq = np.linspace(min(x_clean), max(x_clean), num=100)
        yseq = a + b * xseq
        ax.plot(xseq, yseq, color=color, lw=1.5, alpha=0.6, linestyle='--')
        equation = f"{label}\ny = {b:.2f}x + {a:.2f}"
        ax.text(x_text, y_text, equation, color=color, fontsize=30, ha='center')


def plot_linear_fit_with_text_on_line(ax, x, y, color, text, x_text, y_text):
    x_clean, y_clean = filter_nan(x, y)
    if len(x_clean) > 1:
        b, a = np.polyfit(x_clean, y_clean, deg=1)
        xseq = np.linspace(min(x_clean), max(x_clean), num=100)
        yseq = a + b * xseq
        angle = np.degrees(np.arctan(b))
        ax.plot(xseq, yseq, color=color, lw=1.5, alpha=0.6, linestyle='--')
        ax.text(x_text, y_text, text, color=color, fontsize=24, ha='center', va='center',
                alpha=0.75, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)


def plot_line_with_text(ax, x_start, x_end, slope, intercept, color, text):
    x = np.linspace(x_start, x_end, num=100)
    y = intercept + slope * x
    angle = np.degrees(np.arctan(slope))
    x_mid = (x_start + x_end) / 1.26
    y_mid = intercept + slope * x_mid - 1.67
    ax.plot(x, y, color=color, lw=2.0, alpha=0.6)
    ax.text(x_mid, y_mid, text, color=color, fontsize=24, ha='center', va='center',
            alpha=0.75, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)


# =====================================
# === Data Processing for miniRUEDI ===
# =====================================

def load_csv(file_path, sep=";"):
    return pd.read_csv(file_path, sep=sep)

def calculate_P_add(df, gases):
    return df[gases].sum(axis=1)

def convert_pressure(pressure):
    return np.where(pressure < 2.5, pressure * 1000, pressure)

def process_dataframe(df, gases):
    df["P_add"] = calculate_P_add(df, gases)
    df["P"] = convert_pressure(df["P"])
    df["# Time"] = pd.to_datetime((df["# Time"].values * 1e9).astype(int))
    df.set_index("# Time", inplace=True)
    df_norm = df[gases].multiply(df["P"], axis="index").div(df["P_add"], axis="index")
    df_norm = pd.concat([df_norm, df[["P", "T"]]], axis=1)
    return df_norm


def compute_limits(df, gas):
    mean, std = df[gas].mean(), df[gas].std()
    return mean, std, mean - 3 * std, mean + 3 * std

def trim_outliers(df, gas, lower_limit, upper_limit):
    return df[(df[gas] > lower_limit) & (df[gas] < upper_limit)]

def date_filter(df, start_date, end_date):
    return df[(df.index < start_date) | (df.index > end_date)]

def remove_rows_by_condition(dfs_norm, well, date_1, date_2, gas, value, direction):
    if direction not in ('<', '>'):
        raise ValueError("Direction must be '<' or '>'")
    condition = (dfs_norm[well].index >= date_1) & (dfs_norm[well].index <= date_2)
    condition &= dfs_norm[well][gas] < value if direction == '<' else dfs_norm[well][gas] > value
    dfs_norm[well] = dfs_norm[well].loc[~condition]


# =====================================
# === Visualization & Resampling ===
# =====================================

def plot_distribution(ax, df, gas, mean, std, lower_limit, upper_limit):
    ax.hist(df[gas], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(mean, color='r', linestyle='-', label='Mean')
    ax.axvline(upper_limit, color='r', linestyle='--', label='+3σ')
    ax.axvline(lower_limit, color='r', linestyle='--', label='-3σ')
    ax.set_title(f"Distribution of {gas}")
    ax.legend()

def setup_subplots(rows=2, cols=2):
    return plt.subplots(rows, cols, figsize=(16, 10))

def resample_data(df, well='NA'):
    df_h, df_d = df.resample('15min').mean(), df.resample('1d').mean()
    if well != 'NA':
        df_h['Well'], df_d['Well'] = well, well
    return df_h, df_d

def resample_data_dict(dfs_norm):
    dfs_h, dfs_d = {}, {}
    for well, df in dfs_norm.items():
        well_id = ''.join(filter(str.isdigit, well))
        dfs_h[well], dfs_d[well] = resample_data(df, well=well_id)
    return dfs_h, dfs_d


# =====================================
# === Sinusoidal Fitting ===
# =====================================

def sinusoidal_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def fitter(x, x_real, y, initial_params):
    bounds = ([0, 2*np.pi/365 - 1e-6, -np.inf, -np.inf], [np.inf, 2*np.pi/365 + 1e-6, np.inf, np.inf])
    params, _ = curve_fit(sinusoidal_func, x, y, p0=initial_params, bounds=bounds)
    y_fit = sinusoidal_func(x, *params)
    r_squared = r2_score(y, y_fit)
    y_plot = sinusoidal_func(mdates.date2num(x_real), *params)
    y_error = y - y_fit
    df_prederror = pd.DataFrame({"Fitted_Value": y_fit, "Predicted_Error": y_error}, index=mdates.num2date(x))
    df_prederror = df_prederror.resample('D').asfreq()
    return df_prederror, y_fit, y_plot, r_squared
