# pymlogit
Estimation of discrete choice models in python

### Example:
The following example analyzes choices of fishing locations. For information about the data see: https://doi.org/10.1162/003465399767923827
The data is imported with pandas and then a multinomial model is fitted by passing numpy arrays X and y. The name of the alternatives need to be passed in order to create the output names and in order to know how the data in long format is splitted across alternatives

#### Usage
```python
import pandas as pd
from pymlogit.linear import MultinomialModel

df = pd.read_csv("examples/data/fishing_long.csv") #Data needs to be in long format

X = df[['income','price']].values 
y = df['choice'].values #Choices represented with zeros and ones

model = MultinomialModel()
model.fit(X,y,varnames = ['income','price'],isvars = ['income'], asvars=['price'],alternatives=['beach','boat','charter','pier'])
model.summary()
```

#### Output
```
Optimization succesfully completed after 11 iterations. 
-----------------------------------------------------------------------------------------
Coefficient             Estimate        Std. Error      z-value         Pr(>|z|)     
-----------------------------------------------------------------------------------------
_intercept.boat         0.4928935957    0.2053370982    2.4004118111    0.0449401617 .    
_intercept.charter      1.8540668405    0.2097451458    8.8396173995    0.0000000000 ***  
_intercept.pier         0.7526662342    0.2042533633    3.6849637242    0.0009279897 **   
income.boat             0.0000933295    0.0000471101    1.9810953073    0.1122778277      
income.charter          -0.0000324867   0.0000478462    -0.6789828929   0.6333945307      
income.pier             -0.0001267191   0.0000465724    -2.7209078124   0.0198559789 .    
price                   -0.0255642838   0.0015153615    -16.8700891463  0.0000000000 ***  
-----------------------------------------------------------------------------------------
Significance:  *** 0    ** 0.001    * 0.01    . 0.05

Log-Likelihood= -1220.535
```

### Installation
Download source code and import pymlogit.linear.MultinomialModel

Notes:
The current version allows estimation of:
- Multinomial Logit Models: Models with individual specific variables
- Conditional Logit Models: Models with alternative specific variables
- Models with both, individual and alternative specific variables

The current version does not support models with panel data.
