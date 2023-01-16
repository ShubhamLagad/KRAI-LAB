# 10. SVM classification on any dataset


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')
print("Dataframe : \n\n",df.head())

x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=91)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lin = SVC(kernel='linear')
model_lin.fit(X_train_scaled, y_train)
print("Linear score : ", model_lin.score(X_test_scaled, y_test))

model_poly = SVC(kernel='poly')
model_poly.fit(X_train_scaled, y_train)
print("Poly score : ", model_poly.score(X_test_scaled, y_test))

model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train_scaled, y_train)
print("RBF score : ", model_rbf.score(X_test_scaled, y_test))

class_0_act = X_test[y_test == 0]
class_1_act = X_test[y_test == 1]

plt.scatter(class_0_act['Age'], class_0_act['EstimatedSalary'], c='red')
plt.scatter(class_1_act['Age'], class_1_act['EstimatedSalary'], c='blue')
plt.show()

y_pre = model_lin.predict(X_test_scaled)
class_0_pre = X_test[y_pre == 0]
class_1_pre = X_test[y_pre == 1]

plt.scatter(class_0_pre['Age'], class_0_pre['EstimatedSalary'], c='red')
plt.scatter(class_1_pre['Age'], class_1_pre['EstimatedSalary'], c='blue')
plt.show()

y_pre = model_poly.predict(X_test_scaled)
class_0_pre = X_test[y_pre == 0]
class_1_pre = X_test[y_pre == 1]

plt.scatter(class_0_pre['Age'], class_0_pre['EstimatedSalary'], c='red')
plt.scatter(class_1_pre['Age'], class_1_pre['EstimatedSalary'], c='blue')
plt.show()

y_pre = model_rbf.predict(X_test_scaled)
class_0_pre = X_test[y_pre == 0]
class_1_pre = X_test[y_pre == 1]

plt.scatter(class_0_pre['Age'], class_0_pre['EstimatedSalary'], c='red')
plt.scatter(class_1_pre['Age'], class_1_pre['EstimatedSalary'], c='blue')
plt.show()


# OUTPUT

# Dataframe : 

#      User ID  Gender  Age  EstimatedSalary  Purchased
# 0  15624510    Male   19            19000          0 
# 1  15810944    Male   35            20000          0 
# 2  15668575  Female   26            43000          0 
# 3  15603246  Female   27            57000          0 
# 4  15804002    Male   19            76000          0 
# Linear score :  0.79
# Poly score :  0.88  
# RBF score :  0.88
