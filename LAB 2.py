# 2. Plot the correlation plot on dataset and visualize giving an overview of relationships 
# among data on iris data.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

print(df.head())
print(df.shape)

sns.countplot(x='class',data=df)
plt.show()

sns.scatterplot(x='sepallength', y='sepalwidth',
                hue='class', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='petallength', y='petalwidth',
                hue='class', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()



