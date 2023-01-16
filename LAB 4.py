# 4. Apply linear regression Model techniques to predict the data on any dataset

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data_reg.csv')
df.head()

df.shape
plt.scatter(df['YearsExperience'], df['Salary'])
plt.ylim([0, 130000])
model = LinearRegression()
model.fit(df[['YearsExperience']], df['Salary'])

model.predict([[11]])[0]
x1,x2 = 0,11
y1 = model.predict([[x1]])[0]
y2 = model.predict([[x2]])[0]

plt.plot([x1, x2], [y1, y2])
plt.scatter(df['YearsExperience'], df['Salary'])
plt.ylim([0, 130000])
plt.title('Linear Regression')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
