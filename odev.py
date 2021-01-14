import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
veriler=pd.read_csv("tenis.csv")
print(veriler)

# kategorik verilerin işlenmesi
outlook=veriler.iloc[:,0:1].values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
outlook[:,0]=le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

play=veriler.iloc[:,4:5].values
play[:,0]=le.fit_transform(veriler.iloc[:,4:5])
print(play) 

windy=veriler.iloc[:,4:5].values
windy[:,0]=le.fit_transform(veriler.iloc[:,4:5])
print(windy) 
#verilerin birleştirilmesi
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['sunny','overcast','rainy'])
print(sonuc)

veri= veriler.iloc[:,1:3]
sonuc2 = pd.DataFrame(data=veri, index = range(14), columns = ['temperature','humidity'])
print(sonuc2)

sonuc3 = pd.DataFrame(data=play, index = range(14), columns = ['play'])
print(sonuc3)

sonuc4 = pd.DataFrame(data=windy, index = range(14), columns = ['windy'])
print(sonuc4)

s= pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2) 

s3= pd.concat([s2,sonuc4],axis=1)
print(s3) 

#veri kümesinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

# öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

# basit doğrusal regresyon
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test)
tahmin=lr.predict(X_test)
print(tahmin)

# basit doğrusal regresyon görselleştirme
X_train=x_train.sort_index()
Y_train=y_train.sort_index()
plt.plot(X_train,Y_train)
plt.plot(X_test,lr.predict(X_test))
# ---------------
# çoklu doğrusal regresyon
plays=s3.iloc[:,5:6].values
print(plays)
sol=s3.iloc[:,:5]
sag=s3.iloc[:,7:8]
enson=pd.concat([sol,sag],axis=1)
#print(enson)

x_train,x_test,y_train,y_test=train_test_split(enson,plays,test_size=0.33,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)

# Geriye doğru eleme yöntemi
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=enson,axis=1)
X_l=enson.iloc[:,[0,1,2,3,4]].values 
X_l=np.array(X_l,dtype=float)
model=sm.OLS(plays,X_l).fit()
print(model)
