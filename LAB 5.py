# 5. Apply logical regression Model techniques to predict the data on any dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())

plt.scatter(df['Age'], df['EstimatedSalary'])

df_0 = df[df['Purchased'] == 0]
df_1 = df[df['Purchased'] == 1]

plt.scatter(df_0['Age'], df_0['EstimatedSalary'], c='red')
plt.scatter(df_1['Age'], df_1['EstimatedSalary'], c='blue')

x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

model = LogisticRegression()
model.fit(x, y)
q1 = [[52, 130000], [60, 150000]]
print(model.predict(q1))

scaler = MinMaxScaler()
scaler.fit(x)
X_scaled = scaler.transform(x)

model = LogisticRegression()
model.fit(X_scaled, df['Purchased'])
q1 = [[52, 130000], [60, 150000]]

model.predict(q1)
model.score(X_scaled, df['Purchased'])

plt.show()
