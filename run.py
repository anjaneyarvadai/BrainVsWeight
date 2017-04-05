import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

#read file using pandas
dataframe = pandas.read_fwf('data.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train ML model
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
