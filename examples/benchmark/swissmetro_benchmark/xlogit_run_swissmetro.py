import pandas
import numpy as np
import sys
sys.path.append("../../../")  # Path of xlogit library root folder.

df_wide = pandas.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")
df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) \
                   & (df_wide['CHOICE'] != 0))]  # Filter samples
df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier
df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2: 'SM', 3: 'CAR'})

# ===== STEP 2. RESHAPE DATA TO LONG FORMAT ===== 
from xlogit.utils import wide_to_long
df = wide_to_long(df_wide, id_col='custom_id', alt_name='alt',
                  alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
                  varying=['TT', 'CO', 'AV'], alt_is_prefix=True)

# ===== STEP 3. CREATE MODEL SPECIFICATION ===== 
df['ASC_TRAIN'] = np.ones(len(df))*(df['alt'] == 'TRAIN')
df['ASC_CAR'] = np.ones(len(df))*(df['alt'] == 'CAR')
df['TT'], df['CO'] = df['TT']/100, df['CO']/100  # Scale variables
annual_pass = (df['GA'] == 1) & (df['alt'].isin(['TRAIN', 'SM']))
df.loc[annual_pass, 'CO'] = 0  # Cost zero for pass holders

# ===== STEP 4. ESTIMATE MODEL PARAMETERS ===== 
from xlogit import MixedLogit
varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT']
model = MixedLogit()
model.fit(X=df[varnames], y=df['CHOICE'], varnames=varnames,
          alts=df['alt'], ids=df['custom_id'], panels=df["ID"],
          avail=df['AV'], randvars={'TT': 'n'}, n_draws=1500)
model.summary()
"""
OUTPUT:
Estimation time= 1.3 seconds
---------------------------------------------------------------------------
Coefficient              Estimate      Std.Err.         z-val         P>|z|
---------------------------------------------------------------------------
ASC_CAR                 0.2831101     0.0564250     5.0174585      3.29e-06 ***
ASC_TRAIN              -0.5722759     0.0791059    -7.2343003      7.96e-12 ***
CO                     -1.6601672     0.0778264   -21.3316760      3.38e-78 ***
TT                     -3.2289890     0.1749384   -18.4578595      6.36e-62 ***
sd.TT                   3.6485428     0.1667409    21.8815047       2.2e-81 ***
---------------------------------------------------------------------------
Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood= -4359.218
AIC= 8728.436
BIC= 8751.550
"""
