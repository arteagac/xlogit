# xlogit: A Python package for GPU-accelerated estimation of mixed logit models.

[![Build Status](https://travis-ci.com/arteagac/xlogit.svg?branch=master)](https://travis-ci.com/arteagac/xlogit)
[![Documentation Status](https://readthedocs.org/projects/xlogit/badge/?version=latest)](https://xlogit.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/arteagac/xlogit/badge.svg?branch=master)](https://coveralls.io/github/arteagac/xlogit?branch=master)


### Example:
The following example analyzes choices of fishing modes. See the data [here](examples/data/fishing_long.csv) and more information about the data [here](https://doi.org/10.1162/003465399767923827). The parameters are:
- `X`: Data matrix in long format (numpy array, shape [n_samples, n_variables])
- `y`: Binary vector of choices (numpy array, shape [n_samples, ])
- `varnames`: List of variable names. It must match the number and order of the columns in `X`
- `alts`:  List of alternatives names or codes.
- `randvars`: Variables with random distribution. (`"n"` normal, `"ln"` lognormal, `"t"` triangular, `"u"` uniform, `"tn"` truncated normal)

The current version of `xlogit` only supports data in long format.

#### Usage
```python
# Read data from CSV file
import pandas as pd
df = pd.read_csv("examples/data/fishing_long.csv")

X = df[['price', 'catch']]
y = df['choice']

# Fit the model with xlogit
from xlogit import MixedLogit
model = MixedLogit()
model.fit(X, y,
          varnames=['price', 'catch'],
          ids=df['id'],
          alts=df['alt'],
          randvars={'price': 'n', 'catch': 'n'})
model.summary()
```

#### Output
```
Estimation succesfully completed after 21 iterations.
------------------------------------------------------------------------
Coefficient           Estimate      Std.Err.         z-val         P>|z|
------------------------------------------------------------------------
price               -0.0274061     0.0022827   -12.0062499       2.2e-30 ***
catch                1.3345446     0.1735364     7.6902874      2.29e-13 ***
sd.price             0.0104608     0.0020466     5.1113049      1.93e-06 ***
sd.catch             1.5857201     0.3746104     4.2329844      0.000109 ***
------------------------------------------------------------------------
Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood= -1300.227
AIC= 2608.454
BIC= 2628.754
Estimation time= 0.7 seconds
```
For more examples of `xlogit` see [this Jupyter Notebook](https://github.com/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb).
To test how fast is MixedLogit with GPU processing you can use Google Colaboratory that provides some GPU processing for free. In the Jupyter Notebook above you just need to click the "Open in Colab" button to run your analysis.

## Installation
Install using pip:  
`pip install xlogit`  
Alternatively, you can download source code and import `xlogit.MixedLogit`

### Enable GPU Processsing
To enable GPU processing you must install the CuPy library  ([see installation instructions](https://xlogit.readthedocs.io/en/latest/install.html).  When xlogit detects that CuPy is installed, it switches to GPU processing without any additional setup.

## No GPU? No problem
xlogit also works without a GPU. However, if you need to speed up your model estimation, there are several low cost and even free options to access cloud GPU resources. For instance:

- [Google Colab](https://colab.research.google.com>) offers free GPU resources for learning purposes with no setup required, as the service can be accessed using a web browser. Using xlogit in Google Colab is very easy as it works out of the box without needing to install CUDA or CuPy, which are installed by default. For examples of xlogit running in Google Colab [see this notebook](https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb)
- The [Google Cloud platform](https://cloud.google.com/compute/gpus-pricing) offers GPU processing starting at $0.45 USD per hour for a NVIDIA Tesla K80 GPU with 4,992 CUDA cores.
- [Amazon Sagemaker](https://aws.amazon.com/ec2/instance-types/p2/) offers virtual machine instances with the same TESLA K80 GPU at less than $1 USD per hour.

## Notes:
The current version allows estimation of:
- Mixed logit models with normal, lognormal, triangular, uniform, and truncated normal distributions.
- Mixed logit models with panel data (balanced or unbalanced).
- Multinomial Logit Models: Models with individual specific variables
- Conditional Logit Models: Models with alternative specific variables
- Models with both, individual and alternative specific variables

