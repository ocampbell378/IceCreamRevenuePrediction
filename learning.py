import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from main import sales_df  # Importing the DataFrame from main.py

# Display the first few rows of the DataFrame
print(sales_df.head())

# Display the last few rows of the DataFrame
print(sales_df.tail())

# Display information about the DataFrame
print(sales_df.info())

# Display summary statistics for the DataFrame
print(sales_df.describe())

# Calculate the average temperature and revenue
average_temperature = sales_df['Temperature'].mean()
average_revenue = sales_df['Revenue'].mean()

# Calculate the maximum temperature and revenue
max_temperature = sales_df['Temperature'].max()
max_revenue = sales_df['Revenue'].max()

# Print the calculated values
print("Average Temperature:", average_temperature)
print("Average Revenue:", average_revenue)
print("Max Temperature:", max_temperature)
print("Max Revenue:", max_revenue)
