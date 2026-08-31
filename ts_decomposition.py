# ==============================================================================
# 1. LIBRARY IMPORTS & THEIR PURPOSES
# ==============================================================================
# pandas: Used for data loading, data manipulation, checking dtypes, and indexing.
import pandas as pd

# seaborn: Provides built-in color palettes and high-quality aesthetic themes for plots.
import seaborn as sns

# matplotlib.pyplot: Used to render, customize, and display plots and subplots.
import matplotlib.pyplot as plt

# seasonal_decompose: The statistical function from statsmodels that mathematically 
# splits a time series into Trend, Seasonality, and Residual components.
from statsmodels.tsa.seasonal import seasonal_decompose


# ==============================================================================
# 2. DATA LOADING & INITIAL CHECKS
# ==============================================================================
file_path = "/home/sankalpnaique/PyWorks/BGF/data/automobile_sales.csv"
df = pd.read_csv(file_path)

# Verify data loaded correctly by viewing the first few rows
print("--- First 5 Rows of Dataset ---")
print(df.head())

# Verify data types (confirming 'qtr' is already an integer and 'sales' is numeric)
print("\n--- Data Types of Columns ---")
print(df.dtypes)


# ==============================================================================
# 3. PREPARE DATAFRAME FOR TIME SERIES ANALYSIS
# ==============================================================================
# Set the numeric 'qtr' column directly as the DataFrame index
df.set_index('qtr', inplace=True)

# Ensure the numeric quarters are sorted sequentially (1, 2, 3... N)
df = df.sort_index()


# ==============================================================================
# 4. INITIAL TIME SERIES PLOT (RAW DATA)
# ==============================================================================
# Apply a clean Seaborn visual style
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sales'], color='#1f2937', linewidth=2, marker='o', markersize=4)
plt.title('Automobile Sales Over Time (Raw Time Series)', fontsize=12, fontweight='bold')
plt.xlabel('Quarter (Numeric Index)')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()


# ==============================================================================
# 5. TIME SERIES DECOMPOSITION
# ==============================================================================
# We use an Additive Model: Observed = Trend + Seasonality + Residuals
# period=4 tells statsmodels that the seasonal pattern repeats every 4 numeric quarters (1 year).
decomposition = seasonal_decompose(df['sales'], model='additive', period=4)


# ==============================================================================
# 6. REFINED DECOMPOSITION VISUALIZATION WITH INLINE EXPLANATIONS
# ==============================================================================
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

# --- PANEL 1: OBSERVED ---
# MEANING: The raw, unaltered automobile sales data combining all effects together.
axes[0].plot(df.index, df['sales'], color='#1f2937', linewidth=1.5, label='Observed')
axes[0].set_title('Time Series Decomposition Components', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Observed')

# --- PANEL 2: TREND ---
# MEANING: The long-term direction of sales (overall growth or decline) over time, 
# calculated by smoothing out short-term quarterly fluctuations using a moving average.
axes[1].plot(df.index, decomposition.trend, color='#2563eb', linewidth=2, label='Trend')
axes[1].fill_between(df.index, decomposition.trend, color='#2563eb', alpha=0.1)
axes[1].set_ylabel('Trend')

# --- PANEL 3: SEASONALITY ---
# MEANING: Short-term, regular patterns that repeat predictably every 4 quarters 
# (e.g., consistent sales spikes every Q4 due to end-of-year discounts).
axes[2].plot(df.index, decomposition.seasonal, color='#059669', linewidth=2, label='Seasonal')
axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Seasonal')

# --- PANEL 4: RANDOMNESS / RESIDUALS ---
# MEANING: Unpredictable noise or sudden irregular events remaining after subtracting 
# the Trend and Seasonality from the Observed data (Observed - Trend - Seasonality).
axes[3].scatter(df.index, decomposition.resid, color='#dc2626', s=20, label='Residuals')
axes[3].vlines(df.index, 0, decomposition.resid, color='#dc2626', alpha=0.4, linewidth=1.2)
axes[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Quarter (Numeric Index)')

# Formatting layout clean-up
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white')

plt.tight_layout()
plt.show()