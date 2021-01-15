import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:3]
X=x.values
Y=y.values

#linearRegression uyguladık  önce veride 
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

# LinearRegression görselleştirilmesi
plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
plt.show()

# Polinomal Regresyon  
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
print(x_poly)
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)
# Polinomal Regresyon görselleştirilmesi
plt.scatter(X,Y)
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()