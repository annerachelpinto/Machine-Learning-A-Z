#Random Forest Regression

#Importng the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #to convert to matrix
Y = dataset.iloc[:, 2].values
  
#Splitting the data set into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)'''

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)'''

#Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)

#Predict a new result 
Y_pred = regressor.predict(6.5)

#Visualising the Random Forest Regression results for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01) #its in vector
X_grid = X_grid.reshape((len(X_grid), 1)) #to make it matrix
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
