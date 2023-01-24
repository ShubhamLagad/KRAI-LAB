# 4. Apply linear regression Model techniques to predict the data on any dataset

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data_reg.csv')

plt.scatter(df['YearsExperience'], df['Salary'])
plt.ylim([0, 130000])
model = LinearRegression()
model.fit(df[['YearsExperience']], df['Salary'])

model.predict([[21]])[0]
x1,x2 = 0,21
y1 = model.predict([[x1]])[0]
y2 = model.predict([[x2]])[0]

plt.plot([x1, x2], [y1, y2])
plt.title('Linear Regression')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

# OUTPUT
#    YearsExperience  Salary
# 0              1.1   39343
# 1              1.3   46205
# 2              1.5   37731
# 3              2.0   43525
# 4              2.2   39891






# import numpy as nm  
# import matplotlib.pyplot as mtp  
# import pandas as pd  
# data_set= pd.read_csv('Salary_Data_reg.csv')  
# x= data_set.iloc[:, :-1].values  
# y= data_set.iloc[:, 1].values   
# # Splitting the dataset into training and test set.  
# from sklearn.model_selection import train_test_split  
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)
# #Fitting the Simple Linear Regression model to the training dataset  
# from sklearn.linear_model import LinearRegression  
# regressor= LinearRegression()  
# regressor.fit(x_train, y_train)  
# #Prediction of Test and Training set result  
# y_pred= regressor.predict(x_test)  
# x_pred= regressor.predict(x_train) 
# print(y_pred,"\n",x_pred) 
# mtp.scatter(x_train, y_train, color="green")   
# mtp.plot(x_train, x_pred, color="red")    
# mtp.title("Salary vs Experience (Training Dataset)")  
# mtp.xlabel("Years of Experience")  
# mtp.ylabel("Salary(In Rupees)")  
# mtp.show()   

# #visualizing the Test set results  
# mtp.scatter(x_test, y_test, color="blue")   
# mtp.plot(x_train, x_pred, color="red")    
# mtp.title("Salary vs Experience (Test Dataset)")  
# mtp.xlabel("Years of Experience")  
# mtp.ylabel("Salary(In Rupees)")  
# mtp.show() 