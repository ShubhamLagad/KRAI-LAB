# 8. Developing and implementing Decision Tree model on the dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())

X = df[['Age','EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=91)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

print("Model score : ",model.score(X_test,y_test), model.score(X_train,y_train))

model_1 = DecisionTreeClassifier(max_depth=3,min_samples_leaf=3)
model_1.fit(X_train,y_train)

print("Model score : ",model_1.score(X_test,y_test), model.score(X_train,y_train))

# OUTPUT

#  User ID  Gender  Age  EstimatedSalary  Purchased
# 0  15624510    Male   19            19000          0
# 1  15810944    Male   35            20000          0
# 2  15668575  Female   26            43000          0
# 3  15603246  Female   27            57000          0
# 4  15804002    Male   19            76000          0
# Model score :  0.84 0.9933333333333333
# Model score :  0.88 0.9933333333333333


