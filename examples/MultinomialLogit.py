import pandas as pd
from pymlogit.linear import MultinomialModel

df = pd.read_csv("examples/data/fishing_long.csv")

X = df[['price']].values
y = df['choice'].values

model = MultinomialModel()
model.fit(X,y,varnames = ['income','price'],isvars = ['income'], asvars=['price'],alternatives=['beach','boat','charter','pier'])
model.summary()


model = MultinomialModel()
model.fit(X,y,varnames = ['price'],asvars = ['price'],alternatives=['beach','boat','charter','pier'],fit_intercept=False)
model.summary()
