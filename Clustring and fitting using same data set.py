Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... import numpy as np
... from sklearn.cluster import KMeans
... from sklearn.preprocessing import StandardScaler
... from scipy.optimize import curve_fit
... import matplotlib.pyplot as plt
... 
... # load data
... df = pd.read_csv('data.csv')
... 
... # normalize data
... scaler = StandardScaler()
... df_norm = scaler.fit_transform(df[['GDP per capita', 'CO2 per capita']])
... 
... # cluster data
... kmeans = KMeans(n_clusters=3, random_state=0).fit(df_norm)
... df['Cluster'] = kmeans.labels_
... 
... # pick one country from each cluster
... countries = df.groupby('Cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)
... 
... # compare countries from one cluster
... cluster1 = df[df['Cluster'] == 0][['Country', 'GDP per capita', 'CO2 per capita']]
... cluster1_compare = pd.merge(cluster1, countries, on='Country', how='inner')
... 
... # find similarities and differences
... similarity = cluster1_compare[['Country', 'GDP per capita_x', 'CO2 per capita_x']]
... difference = cluster1_compare[['Country', 'GDP per capita_y', 'CO2 per capita_y']]
... 
... # fit data with a function
... def func(x, a, b):
...     return a * x + b
... 
xdata = df['GDP per capita']
ydata = df['CO2 per capita']

popt, pcov = curve_fit(func, xdata, ydata)

# predict values in ten or twenty years time including confidence ranges
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n = len(ydata)
    dof = max(0, n - len(popt))
    tval = abs(stats.t.ppf(alpha / 2, dof))
    resid = ydata - func(xdata, *popt)
    s_err = np.sqrt(np.sum(resid ** 2) / dof)
    bounds = tval * s_err * np.sqrt(1 + (xdata - np.mean(xdata)) ** 2 / np.sum((xdata - np.mean(xdata)) ** 2))
    return bounds

x = np.linspace(min(xdata), max(xdata), 100)
y = func(x, *popt)
lower, upper = err_ranges(func, xdata, ydata, popt, pcov)

# plot data with the best fitting function and the confidence range
fig, ax = plt.subplots()
ax.plot(xdata, ydata, 'o', label='Data')
ax.plot(x, y, label='Best fit')
ax.fill_between(x, y+upper, y-lower, alpha=0.2, label='Confidence range')
ax.legend()
plt.show()
