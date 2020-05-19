from pymlogit.linear import MultinomialModel

import pandas as pd
df = pd.read_csv("fishing_long.csv")

#============ Estimate model with both alternative specific and individual specific variables
varnames = ['income','price']
X = df[varnames].values
y = df['choice'].values

model = MultinomialModel()
model.fit(X,y,isvars = ['income'], asvars=['price'],alternatives=['beach','boat','charter','pier'],varnames= varnames)
model.summary()

#============ Estimate model with only individual specific variables
varnames = ['income']
X = df[varnames].values
y = df['choice'].values

model = MultinomialModel()
model.fit(X,y,isvars=['income'],alternatives=['beach','boat','charter','pier'],varnames= varnames)
model.summary()

