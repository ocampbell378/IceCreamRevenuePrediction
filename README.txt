=======================================================================
PROJECT NAME: Ice Cream Sales Prediction using Simple Linear Regression
=======================================================================

## OVERVIEW
This project is a simple linear regression model that predicts ice cream sales revenue based on temperature data, 
utilizing Python libraries like Pandas, Numpy, Seaborn, Matplotlib, and Scikit-learn.

## TABLE OF CONTENTS
1. Installation
2. Usage
3. Features
4. Documentation
6. Credits

### Prerequisites
- Python 3.10.9 (this is the version used for development and testing)
- Third-party libraries: `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, 
`jupyterthemes`, `sklearn.model_selection`, `sklearn.linear_model`

### Installation Steps 
1. Clone the repository:
git clone https://github.com/ocampbell378/IceCreamRevenuePrediction.git
2. Install the required libraries:
pip install -r requirements.txt

## USAGE
To run the project, use the following command:
python main.py

## FEATURES
Feature 1: Data Visualization of Temperature vs. Revenue
The script uses Seaborn and Matplotlib to create a scatter plot with a regression line, visualizing the relationship 
between temperature and revenue from an ice cream sales dataset.

Feature 2: Linear Regression Model for Predicting Revenue
The script implements a Simple Linear Regression model using Scikit-learn to predict revenue based on temperature. 
It also calculates the accuracy of the model and provides a predicted revenue for a given temperature.

## DOCUMENTATION
### Modules and Functions

main.py: Handles data visualization, linear regression model creation, and prediction.

import_pandas as pd, import numpy as np, import seaborn as sns, import matplotlib.pyplot as plt: Imports necessary libraries for data manipulation, visualization, and plotting.
jtplot.style(): Applies a custom theme to plots using Jupyter Themes.
pd.read_csv('IceCreamData.csv'): Loads the dataset from a CSV file into a DataFrame.
sns.regplot(): Plots a regression line on the scatter plot of temperature vs. revenue.
train_test_split(): Splits the dataset into training and testing sets for model evaluation.
LinearRegression(): Creates and fits a linear regression model to predict revenue based on temperature.
score(): Evaluates the accuracy of the linear regression model.
predict(): Predicts revenue for a given temperature using the trained model.

learning.py: Contains exploratory data analysis and summary statistics.

print(sales_df.head()): Displays the first few rows of the dataset to get an overview.
print(sales_df.tail()): Displays the last few rows of the dataset.
print(sales_df.info()): Prints information about the dataset, including data types and missing values.
print(sales_df.describe()): Provides summary statistics for the numerical columns in the dataset.
mean(), max(): Calculates and prints the average and maximum temperature and revenue values from the dataset.

## CREDITS
Developed by Owen Campbell
This project was created as part of a guided project from "Simple Linear Regression for the Absolute Beginner" on Coursera.