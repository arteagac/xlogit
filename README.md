# pymlogit
Estimation of multinomial logit models in python 

### Example:
The following example analyzes choices of fishing modes. See the data [here](examples/data/fishing_long.csv) and more information about the data [here](https://doi.org/10.1162/003465399767923827). The parameters are:
- `X`: Data matrix in long format (numpy array, shape [n_samples, n_fvariables])
- `y`: Binary vector of choices (numpy array, shape [n_samples, ])
- `varnames`: List of variable names. Its length must match number of columns in `X`
- `alternatives`:  List of alternatives names or codes.
- `asvars`: List of alternative specific variables
- `isvars`: List of individual specific variables
The current version of `pymlogit` only supports data in long format.

#### Usage
```python
from pymlogit.linear import MultinomialModel

import pandas as pd
df = pd.read_csv("examples/data/fishing_long.csv")

varnames = ['income','price']
X = df[varnames].values
y = df['choice'].values

model = MultinomialModel()
model.fit(X,y,isvars = ['income'], asvars=['price'],alternatives=['beach','boat','charter','pier'],varnames= varnames)
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
Install using pip:  
`pip install pymlogit`  
Alternatively, you can download source code and import pymlogit.linear.MultinomialModel

### Notes:
The current version allows estimation of:
- Multinomial Logit Models: Models with individual specific variables
- Conditional Logit Models: Models with alternative specific variables
- Models with both, individual and alternative specific variables

The current version does not support models with panel data.
