Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
... import matplotlib.pyplot as plt
... from scipy.optimize import curve_fit
... 
... def exponential_growth(t, a, b):
...     return a * np.exp(b * t)
... 
... x = np.array([0, 5, 10, 15, 20, 25, 30])
... y = np.array([5.0, 6.2, 7.9, 10.1, 13.0, 16.6, 21.2])
... 
... popt, pcov = curve_fit(exponential_growth, x, y, p0=(1.0, 0.1))
... print("Fitted parameters:", popt)
... 
... # Predictions for 10 and 20 years
... x_pred = np.array([40, 50])
... y_pred = exponential_growth(x_pred, *popt)
... print("Predicted values:", y_pred)
... 
... # Confidence range
... sigma = np.sqrt(np.diag(pcov))
... conf_range = err_ranges(x_pred, y_pred, sigma, exponential_growth)
... 
... # Plotting the data and the fitted function
... plt.plot(x, y, 'bo', label="Data")
... plt.plot(x_pred, y_pred, 'r-', label="Fitted function")
... plt.fill_between(x_pred, conf_range[0], conf_range[1], alpha=0.2, color='r', label='Confidence Range')
... plt.xlabel("Years")
... plt.ylabel("CO2 Emissions (Gigatons)")
... plt.legend(loc="best")
... plt.show()
