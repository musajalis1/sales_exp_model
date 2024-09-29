# -*- coding: utf-8 -*-
"""
# Modeling a sales dataset with an exponential function

In this notebook, I will be walking you through modeling data with an exponential function. The dataset provides us with sales orders over a period of a few years. The dataset contains information regarding the date of an order, the quantity of orders, the unit price, and other identifying information for each order.

For the purpose of this notebook, we will be taking a closer look at the quantity ordered and the period of time in which the orders take place. We will be using this data to build a model for total sales, that can be used to predict future sales.

We will be modelling the data using the function: y = a_fit * e^(b_fit * x)

The dataset used can be found at: https://github.com/MicrosoftLearning/dp-data/tree/main

# The Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy
import numpy as np

"""We first pull the csv file into the notebook from GitHub. Next, we can take a look at the dataframe we have created from the data."""

data_src = "https://raw.githubusercontent.com/MicrosoftLearning/dp-data/refs/heads/main/sales.csv"
sales = pd.read_csv(data_src)

sales

"""We will now create a pivot table to better understand the data. The pivot table is indexed on the date and the values of the table are the aggregate sum of the sale quantity. This will tell us how many total sales were made each day."""

total_sales = sales.pivot_table(index='OrderDate', values='Quantity', aggfunc='sum')
total_sales = total_sales.reset_index()
total_sales

"""We will visualize this data on a scatter plot, to better understand the relationship between the variables and to identify long-term trends."""

plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
plt.scatter(total_sales['OrderDate'], total_sales['Quantity'], s= 3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Total Sales')

"""From the visualization, we can see that the sales orders have increased over the time period. Furthermore, we can observe an exponential trend that the data follows.

#The Model

Based upon our previous discovery, we will be modeling the data with an exponential function. First, we define the function, which will include the model parameters (a and b) that can be fit to the data.
"""

def exp_func(x, a, b):
  return a * np.exp(b * x)

"""Next we will reformat our date values into integers. The reformatted date column will now represent the days since the first datapoint. This will allow the model to process the date as an independent variable."""

total_sales_dates = pd.to_datetime(total_sales['OrderDate'].tolist())
total_sales_ref_date = total_sales_dates.min()
total_sales_day_count = (total_sales_dates - total_sales_ref_date).days

"""Now, we will fit our model to the data. We are providing the curve_fit() function with the following parameters: the function we are using for the model (exponential function), the x data (days past), the y data (number of sales), and a predicted value for the parameters of the exponential function (a and b). These parameters were reached through a few rounds of trial and error.

Finally, we will extract from the function our fitted model parameters, and the covariance matrix for the fitted parameters. For the purpose of this notebook, we will not be using the covariance matrix.
"""

opt, cov = scipy.optimize.curve_fit(exp_func, total_sales_day_count, total_sales['Quantity'], p0= [0.005, 0.005])

"""We then extract our a and b parameters that have been fitted to the data."""

a_fit, b_fit = opt
print(f"a = {a_fit}, b = {b_fit}")

"""And, finally, we overlay the exponential model over the data. This model can be used to make predictions for future sale quantities.

The data can be modeled with the function: y = a_fit * e^(b_fit * x)
"""

plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
plt.scatter(total_sales['OrderDate'], total_sales['Quantity'], s= 3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

x_range_q = np.linspace(min(total_sales_day_count),
                        max(total_sales_day_count), 100)
y_fit = exp_func(x_range_q, *opt)
plt.plot(x_range_q, y_fit, color= 'red')

plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Total Sales Modeled Exponentially')
