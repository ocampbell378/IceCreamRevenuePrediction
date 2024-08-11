import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sales_df = pd.read_csv('IceCreamData.csv')

plt.figure(figsize= (13, 7))
sns.regplot(x = 'Temperature', y = 'Revenue', data = sales_df)
# plt.show()

X = sales_df['Temperature']
Y = sales_df['Revenue']

X = np.array(X)
Y = np.array(Y)

X = X.reshape(-1,1)

Y = Y.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

SimpleLinearRegression = LinearRegression(fit_intercept = True)
SimpleLinearRegression.fit(X_train, Y_train)

# print('Linear Model Coeff(m)', SimpleLinearRegression.coef_)
# print('Linear Model Coeff(b)', SimpleLinearRegression.intercept_)

plt.scatter(X_test, Y_test, color = 'gray')
plt.plot(X_test, SimpleLinearRegression.predict(X_test), color = 'r')
plt.ylabel('Revenue [$]')
plt.xlabel('Temp. [degC]')
plt.title('Revenue Generated vs. Temperature')
# plt.show()

accuary_LinearRegresssion = SimpleLinearRegression.score(X_test, Y_test)
# print('Accuracy of Linear Regression Model: ', round(accuary_LinearRegresssion * 100, 1), '%')

Temp = np.array([20])
Temp = Temp.reshape(-1,1)

Revenue = SimpleLinearRegression.predict(Temp)
# print('Revenue Predictions =', Revenue)
