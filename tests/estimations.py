import numpy as np
import pandas as pd
import sys
sys.path.append(".")  # Path of xlogit library root folder.

from xlogit import MixedLogit


print("""
**EXPECTED:
convergence=True LL=-3891.7177135708052 electricity
convergence=True LL=-2278.8977801007272 artificial
convergence=True LL=-1300.5113418149986 fishing
convergence=True LL=-1300.5113418149986 fishing batch
pred pier base: 0.089 updated: 0.105
pred pier base: 0.089 updated: 0.105 batch""")

print("**OBTAINED:")
# Electricity dataset
df = pd.read_csv("examples/data/electricity_long.csv")
varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
model = MixedLogit()
model.fit(X=df[varnames],
          y=df['choice'],
          varnames=varnames,
          ids=df['chid'],
          panels=df['id'],
          alts=df['alt'],
          n_draws=500,
          verbose=0,
          randvars={'pf': 'n', 'cl': 'n', 'loc': 'n',
                    'wk': 'n', 'tod': 'n', 'seas': 'n'})
print(f"convergence={model.convergence} LL={model.loglikelihood} electricity")


# Artificial dataset
df = pd.read_csv("examples/data/artificial_long.csv")
varnames = ['price', 'time', 'conven', 'comfort', 'meals', 'petfr',
            'emipp', 'nonsig1', 'nonsig2', 'nonsig3']

model = MixedLogit()
model.fit(X=df[varnames],
          y=df['choice'],
          varnames=varnames,
          alts=df['alt'],
          ids=df['id'],
          n_draws=500,
          panels=None,
          verbose=0,
          randvars={'meals': 'n', 'petfr': 'n', 'emipp': 'n'}
          )

print(f"convergence={model.convergence} LL={model.loglikelihood} artificial")



# Fishing dataset regular and batch
df = pd.read_csv("examples/data/fishing_long.csv")

varnames = ['price', 'catch']
model = MixedLogit()
model.fit(X=df[varnames], y=df['choice'], varnames=varnames, alts=df['alt'],
          ids=df['id'], n_draws=1000, randvars={'price': 'n', 'catch': 'n'}, verbose=0)
print(f"convergence={model.convergence} LL={model.loglikelihood} fishing")


varnames = ['price', 'catch']
model = MixedLogit()
model.fit(X=df[varnames], y=df['choice'], varnames=varnames, alts=df['alt'],
          ids=df['id'], n_draws=1000, randvars={'price': 'n', 'catch': 'n'}, verbose=0, batch_size=200)
print(f"convergence={model.convergence} LL={model.loglikelihood} fishing batch")

# Predict regular and batch
choices, freq = model.predict(X=df[varnames], varnames=varnames, ids=df['id'],
                              alts=df['alt'], return_freq=True, n_draws=1000, verbose=0)

df.loc[df['alt']=='boat', 'price'] *= 1.2  # 20 percent price increase
choices, freq_2 = model.predict(X=df[varnames], varnames=varnames, ids=df['id'],
                              alts=df['alt'], return_freq=True, n_draws=1000, verbose=0)
print(f"pred pier base: {freq['pier']} updated: {freq_2['pier']}")

df = pd.read_csv("examples/data/fishing_long.csv")

choices, freq = model.predict(X=df[varnames], varnames=varnames, ids=df['id'],
                              alts=df['alt'], return_freq=True, n_draws=1000, verbose=0, batch_size=200)

df.loc[df['alt']=='boat', 'price'] *= 1.2  # 20 percent price increase
choices, freq_2 = model.predict(X=df[varnames], varnames=varnames, ids=df['id'],
                              alts=df['alt'], return_freq=True, n_draws=1000, verbose=0, batch_size=200)
print(f"pred pier base: {freq['pier']} updated: {freq_2['pier']} batch")