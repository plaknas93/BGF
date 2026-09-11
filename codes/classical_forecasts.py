
#### Data Generating Process

### Example: Study Hours and Examination Scores

## Data Display and Scatter Plot

import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset

df = pd.DataFrame({
    "Student": range(1, 11),
    "Study_Hours": [1.5, 2.0, 3.5, 4.0, 5.5,
                    6.0, 7.5, 8.0, 9.5, 10.0],
    "Actual_Score": [43, 47, 56, 59, 68,
                     70, 78, 81, 87, 89]
})

# Display the data

display(df)

# Create scatter plot

plt.figure(figsize=(9, 5.5))

plt.scatter(
    df["Study_Hours"],
    df["Actual_Score"],
    s=80
)

plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Actual Score", fontsize=12)
plt.title(
    "Relationship Between Study Hours and Exam Scores",
    fontsize=14,
    pad=15
)

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

## End of Data Display and Scatter Plot Code

## Estimate the Model

import statsmodels.api as sm

# Define dependent and independent variables

X = df["Study_Hours"]
y = df["Actual_Score"]

# Add intercept

X = sm.add_constant(X)

# Estimate the regression model

model = sm.OLS(y, X).fit()

# Display regression output

print(model.summary())

## End of Estimate the Model Code

## Regression Line Plot

import numpy as np
import matplotlib.pyplot as plt

# Generate fitted values

x = np.linspace(
    df["Study_Hours"].min(),
    df["Study_Hours"].max(),
    100
)

b0 = model.params["const"]
b1 = model.params["Study_Hours"]

y_hat = b0 + b1 * x

# Create the plot

plt.figure(figsize=(9, 5.5))

plt.scatter(
    df["Study_Hours"],
    df["Actual_Score"],
    s=80,
    label="Observed data"
)

plt.plot(
    x,
    y_hat,
    linewidth=2.5,
    label="Estimated signal"
)

# Regression equation

equation = f"Score = {b0:.2f} + {b1:.2f} × Study Hours"

plt.text(
    0.05,
    0.93,
    equation,
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray"
    )
)

plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Actual Score", fontsize=12)
plt.title(
    "Observed Scores and the Estimated Signal",
    fontsize=14,
    pad=15
)

plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()

## End of Regression Line Plot Code

## Naive Forecasting Method

import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset
df = pd.DataFrame(
    {
        "Month": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
        ],
        "Stock_Price": [
            1842.50,
            1875.25,
            1821.75,
            1906.50,
            1958.25,
            1932.75,
            2014.50,
            2067.25,
            2041.50,
        ],
    }
)

# Plot monthly stock prices
plt.figure(figsize=(10, 5))

plt.plot(df["Month"], df["Stock_Price"], marker="o")

plt.xlabel("Month")
plt.ylabel("Month-End Stock Price (₹)")
plt.title("Monthly Stock Price")

plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## End of Naive Forecasting Method Code

## Seasonal Naïve Forecasting Method

import pandas as pd
import matplotlib.pyplot as plt
  
# Create the dataset
df = pd.DataFrame({
    "Year": [2024]*4 + [2025]*4,
    "Quarter": ["Q1", "Q2", "Q3", "Q4"] * 2,
    "Sales": [100, 120, 105, 145, 98, 122, 103, 148]
})
  
# Create a sequential time-period variable
df["Period"] = range(1, len(df) + 1)
  
# Plot quarterly sales
plt.figure(figsize=(10, 5))
  
plt.plot(
    df["Period"],
    df["Sales"],
    marker="o"
)
  
plt.xticks(
    df["Period"],
    [f"{year} {quarter}" for year, quarter in zip(df["Year"], df["Quarter"])]
)
  
plt.xlabel("Quarter")
plt.ylabel("Sales (₹ lakh)")
plt.title("Quarterly Sales: 2024–2025")
  
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## End of Seasonal Naïve Forecasting Method Code


## Moving Average Forecasting Method

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/home/sankalpnaique/PyWorks/BGF/data/automobile_sales.csv"
df = pd.read_csv(file_path)

# Inspect the dataset
print("First 5 rows of the dataset:")
display(df.head())

print("\nData types:")
display(df.dtypes)

print("\nNumber of observations and variables:")
print(df.shape)

# Calculate 3-period and 10-period moving averages
df["MA_3"] = df["sales"].rolling(window=3).mean()
df["MA_10"] = df["sales"].rolling(window=10).mean()

# --------------------------------------------------
# Figure 1: Raw quarterly sales
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["qtr"],
    df["sales"],
    linewidth=1.5,
    label="Actual Sales"
)

plt.title(
    "Quarterly Automobile Sales",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("Quarter")
plt.ylabel("Automobile Sales")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Figure 2: Sales with moving averages
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["qtr"],
    df["sales"],
    linewidth=1.2,
    alpha=0.6,
    label="Actual Sales"
)

plt.plot(
    df["qtr"],
    df["MA_3"],
    linewidth=2,
    label="3-Period MA"
)

plt.plot(
    df["qtr"],
    df["MA_10"],
    linewidth=2,
    label="10-Period MA"
)

plt.title(
    "Quarterly Automobile Sales: 3-Period vs. 10-Period Moving Averages",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("Quarter")
plt.ylabel("Automobile Sales")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

## End of Code Moving Average Forecasting