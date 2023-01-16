# 3. Analysis of covariance: variance (ANOVA), if data have categorical variables on iris data.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
from statsmodels.graphics.factorplots import interaction_plot

df = pd.read_csv("iris.csv")

iris = load_iris()

df_iris = pd.DataFrame(df,columns=['ID','sepallength', 'sepalwidth','petallength','petalwidth'])
print("Shape : ",df_iris.shape)


df_iris1 = pd.DataFrame(iris.target,columns=['target'])

df_iris_new = pd.concat([df_iris, df_iris1], axis=1)
print("New Shape : ",df_iris1.shape)

scatter_matrix(df_iris[['sepallength', 'sepalwidth','petallength','petalwidth']],figsize=(15,10))
plt.show()

fig = interaction_plot(df_iris_new['sepalwidth'],df_iris_new['target'],df_iris_new['ID'],colors=['red','blue','green'],ms=12)
plt.show()

print("Dataframe info : \n",df_iris_new.info())
print("Dataframe describe : \n",df_iris_new.describe())


# OUTPUT

# Shape :  (150, 5)    
# New Shape :  (150, 1)
# <class 'pandas.core.frame.DataFrame'>    
# RangeIndex: 150 entries, 0 to 149        
# Data columns (total 6 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   ID           150 non-null    int64  
#  1   sepallength  150 non-null    float64
#  2   sepalwidth   150 non-null    float64
#  3   petallength  150 non-null    float64
#  4   petalwidth   150 non-null    float64
#  5   target       150 non-null    int32
# dtypes: float64(4), int32(1), int64(1)
# memory usage: 6.6 KB
# Dataframe info : 
#  None
# Dataframe describe : 
#                 ID  sepallength  sepalwidth  petallength  petalwidth      target
# count  150.000000   150.000000  150.000000   150.000000  150.000000  150.000000
# mean    75.500000     5.843333    3.054000     3.758667    1.198667    1.000000
# std     43.445368     0.828066    0.433594     1.764420    0.763161    0.819232
# min      1.000000     4.300000    2.000000     1.000000    0.100000    0.000000
# 25%     38.250000     5.100000    2.800000     1.600000    0.300000    0.000000
# 50%     75.500000     5.800000    3.000000     4.350000    1.300000    1.000000
# 75%    112.750000     6.400000    3.300000     5.100000    1.800000    2.000000
# max    150.000000     7.900000    4.400000     6.900000    2.500000    2.000000
