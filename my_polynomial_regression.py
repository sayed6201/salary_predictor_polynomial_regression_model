# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#X = dataset.iloc[:,1:2] upper bound(2) is excluded, only 1 will be in matrix, it'will consider X as a matrix[row,column] not as vector[has only lines]
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#our dataset is samll, 10 lines, 1 columns, so we don't need to divide into trainning set/test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
#not required as Regressor library will take care
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#---------------------------------Step 1---------------------------------


#fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
#2 onjects required, 1st one is for lin_reg,poly_reg
lin_reg = LinearRegression();
#fitting linear regression to our dataset
lin_reg.fit(X,y)


#fitting polynomial regression to the dataset
#PolynomialFeatures will add column of x^n(x^2,x^3..)
from sklearn.preprocessing import PolynomialFeatures
#degree = 2(will have less acurate),degree = 4(the actual prediction)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#now independent variables has increased, x,x^2....
#linear regression of this varibales will be polynomial regression....
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


#---------------------------------Step 2---------------------------------



#visualise the linear regression model
plt.scatter(X , y , color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth ot Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#if you observe the prediction line, you will see datas are too far, few are on line..


#visualise the polynomial linear regression model
#np.arange(min(X),max(X),0.1) will give us vector, min(X) to max(X) with 0.1 increement
X_grid = np.arange(min(X),max(X),0.1)
#X_grid.reshape((len(X_grid), 1)) will create matrix of length(X_grid), and 1 column
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X , y , color = 'red')

#X_poly = poly_reg.fit_transform(X_grid), we are trying to make it more dynamic, that's why not using X_poly
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth ot Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
ply.show()


#-----------------Step 3--------------------


#predicting salaries with linear regression model
#lin_reg.predict(X) ,here X will return corresponding 10 values of y, 6.5 will return corresponsing y only
lin_reg.predict(6.5)


#predicting new result with polynomial regression model
lin_reg2.predict(poly_reg.fit_transform(6.5))