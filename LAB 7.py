# 7.Association algorithms for supervised classification on any dataset 

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())

x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=91)

print(x_train)
print(y_train)

# OUTPUT

# User ID  Gender  Age  EstimatedSalary  Purchased
# 0  15624510    Male   19            19000          0
# 1  15810944    Male   35            20000          0
# 2  15668575  Female   26            43000          0
# 3  15603246  Female   27            57000          0
# 4  15804002    Male   19            76000          0
#      Age  EstimatedSalary
# 168   29           148000
# 144   34            25000
# 336   58           144000
# 56    23            48000
# 344   47           105000
# ..   ...              ...
# 349   38            61000
# 210   48            96000
# 362   47            50000
# 174   34            72000
# 178   24            23000

# [300 rows x 2 columns]
# 168    1
# 144    0
# 336    1
# 56     0
# 344    1
#       ..
# 349    0
# 210    1
# 362    1
# 174    0
# 178    0
# Name: Purchased, Length: 300, dtype: int64