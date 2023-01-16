# 1. Find the correlation matrix

import pandas as pd
data = {
    'x':[25,65,9,85,96],
    'y':[38,69,45,12,75],
    'z':[20,50,12,46,53]
}

df = pd.DataFrame(data,columns=['x','y','z'])
print("DataFrame : \n",df)
print("Correlation matrix : \n",df.corr())

# OUTPUT

# DataFrame : 
#      x   y   z       
# 0  25  38  20        
# 1  65  69  50        
# 2   9  45  12        
# 3  85  12  46        
# 4  96  75  53        
# Correlation matrix : 
#            x         y         z
# x  1.000000  0.176580  0.957816 
# y  0.176580  1.000000  0.327335 
# z  0.957816  0.327335  1.000000