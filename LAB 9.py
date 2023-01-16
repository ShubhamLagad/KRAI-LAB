# 9. Bayesian classification on any dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("adult.csv")

# print(df.head)

# print("Dataframe Infor : \n\n",df.info)
# print("Workclass unique : \n\n",df['workclass'].unique())
# print("Workclass value count : \n\n",df['workclass'].value_counts())
# print(df['occupation'].unique())
# print(df['occupation'].value_counts())
# print(df['native.country'].unique())
# print(df['native.country'].value_counts())

x = df.drop(['income'], axis=1)
y = df['income']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital.status',
                           'occupation', 'relationship', 'race', 'gender', 'native.country'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

print("X_train data : \n", x_train.head())
print("X_train shape : \n", x_train.shape)

print("X_test data : \n", x_train.head())
print("X_test shape : \n", x_train.shape)

col = x_train.columns

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = pd.DataFrame(x_train, columns=[col])
x_test = pd.DataFrame(x_test, columns=[col])
print(x_train.head())

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("Prediction : ", y_pred)


# OUTPUT

# X_train data : 

#      workclass_1  workclass_2  workclass_3  workclass_4  ...  native.country_39  native.country_40  native.country_41  native.country_42
#32098       1            0            0            0      ...             0                  0                 0                  0 
#25206       0            1            0            0      ...             0                  0                  0                  0 
#23491       0            0            1            0      ...             0                  0                  0                  0 
#12367       0            1            0            0      ...             0                  0                  0                  0 
#7054        0            0            0            1      ...             0                  0                  0                  0 

# [5 rows x 103 columns]

# X_train shape : 
#  (22792, 103)   


# X_test data :   

#      workclass_1  workclass_2  workclass_3  workclass_4  ...  native.country_39  native.country_40  native.country_41  native.country_42
#32098          1            0            0            0   ...                 0                  0                  0                  0 
#25206          0            1            0            0   ...                 0                  0                  0                  0 
#23491          0            0            1            0   ...                 0                  0                  0                  0 
#12367          0            1            0            0   ...                 0                  0                  0                  0 
#7054           0            0            0            1   ...                 0                  0                  0                  0 

# [5 rows x 103 columns]

# X_test shape :
#  (22792, 103)

# workclass_1 workclass_2 workclass_3 workclass_4 workclass_5  ... native.country_38 native.country_39 native.country_40 native.country_41 native.country_42
# 0         1.0         0.0        -1.0         0.0         0.0  ...               0.0               0.0               0.0               0.0               0.0   
# 1         0.0         1.0        -1.0         0.0         0.0  ...               0.0               0.0               0.0               0.0               0.0   
# 2         0.0         0.0         0.0         0.0         0.0  ...               0.0               0.0               0.0               0.0               0.0   
# 3         0.0         1.0        -1.0         0.0         0.0  ...               0.0               0.0               0.0               0.0               0.0   
# 4         0.0         0.0        -1.0         1.0         0.0  ...               0.0               0.0               0.0               0.0               0.0   

# [5 rows x 103 columns]

# Prediction :  ['<=50K' '<=50K' '<=50K' ... '<=50K' '<=50K' '>50K']
