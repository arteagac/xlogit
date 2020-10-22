"""
This code estimates a mixed logit model for the artificial
dataset using biogeme. To use this code you must first 
install biogeme by executing:

pip install biogeme

"""

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo

data_folder = "../data/"
df = pd.read_csv(data_folder+"artificial_wide.csv")
df['choice'] = df['choice'].astype('str')
mapping = {'1': 1, '2': 2, '3': 3}

for k, v in mapping.items():
    df["aval_"+k] = np.ones(df.shape[0])

df = df.replace({'choice': mapping})
database = db.Database('artificial', df)

globals().update(database.variables)

# Fixed params
b_price = Beta('b_price', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_conven = Beta('b_conven', 0, None, None, 0)
b_comfort = Beta('b_comfort', 0, None, None, 0)
b_nonsig1 = Beta('b_nonsig1', 0, None, None, 0)
b_nonsig2 = Beta('b_nonsig2', 0, None, None, 0)
b_nonsig3 = Beta('b_nonsig3', 0, None, None, 0)

# Random params
u_meals = Beta('u_meals', 0, None, None, 0)
u_petfr = Beta('u_petfr', 0, None, None, 0)
u_emipp = Beta('u_emipp', 0, None, None, 0)
sd_meals = Beta('sd_meals', 0, None, None, 0)
sd_petfr = Beta('sd_petfr', 0, None, None, 0)
sd_emipp = Beta('sd_emipp', 0, None, None, 0)

b_meals = u_meals + sd_meals*bioDraws('b_meals', 'NORMAL')
b_petfr = u_petfr + sd_petfr*bioDraws('b_petfr', 'NORMAL')
b_emipp = u_emipp + sd_emipp*bioDraws('b_emipp', 'NORMAL')

V1 = price_1*b_price+time_1*b_time+conven_1*b_conven+comfort_1*b_comfort+\
    meals_1*b_meals+petfr_1*b_petfr+emipp_1*b_emipp+nonsig1_1*b_nonsig1+\
        nonsig2_1*b_nonsig2+nonsig3_1*b_nonsig3
V2 = price_2*b_price+time_2*b_time+conven_2*b_conven+comfort_2*b_comfort+\
    meals_2*b_meals+petfr_2*b_petfr+emipp_2*b_emipp+nonsig1_2*b_nonsig1+\
        nonsig2_2*b_nonsig2+nonsig3_2*b_nonsig3
V3 = price_3*b_price+time_3*b_time+conven_3*b_conven+comfort_3*b_comfort+\
    meals_3*b_meals+petfr_3*b_petfr+emipp_3*b_emipp+nonsig1_3*b_nonsig1+\
        nonsig2_3*b_nonsig2+nonsig3_3*b_nonsig3

V = {1: V1, 2: V2, 3: V3}
av = {1: aval_1, 2: aval_2, 3: aval_3}

prob = models.logit(V, av, choice)
logprob = log(MonteCarlo(prob))

# Define level of verbosity
logger = msg.bioMessage()
logger.setGeneral()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=500)
biogeme.modelName = 'MixedLogitArtificial'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)

results.writeHtml()


"""
=============== OUTPUT OF THE ESTIMATION ================
Number of estimated parameters: 	13
Sample size: 	                    4000
Excluded observations:            	0
Init log likelihood: 	           -4394.449
Final log likelihood: 	            -2274.06
Likelihood ratio test for the init. model: 	4240.779
Rho-square for the init. model: 	0.483
Rho-square-bar for the init. model: 	0.48
Akaike Information Criterion:    	4574.119
Bayesian Information Criterion: 	4655.942
Final gradient norm:               1.7917E-02
Number of draws:                   500
Draws generation time:             0:00:03.898370
Types of draws: 	               ['b_emipp: NORMAL', 'b_meals: NORMAL', 'b_petfr: NORMAL']
Nbr of threads: 	               6
Algorithm: 	BFGS with trust region for simple bound constraints
Proportion analytical hessian:     0.0%
Relative projected gradient:       2.794136e-06
Number of iterations: 	           81
Number of function evaluations:    234
Number of gradient evaluations:    77
Number of hessian evaluations:     0
Cause of termination: 	           Relative gradient = 2.8e-06 <= 6.1e-06
Optimization time: 	               0:05:36.622714


Estimated parameters
Name	    Value	Std err	t-test	p-value	Rob. Std err	Rob. t-test	Rob. p-value
b_comfort	1.09	0.202	5.41	6.14e-08	0.202	5.41	6.17e-08
b_conven	0.915	0.161	5.69	1.25e-08	0.163	5.62	1.88e-08
b_nonsig1	0.068	0.149	0.457	0.648	0.145	0.47	0.639
b_nonsig2	-0.00218	0.156	-0.014	0.989	0.149	-0.0146	0.988
b_nonsig3	0.0174	0.137	0.127	0.899	0.134	0.13	0.896
b_price	   -1.06	0.182	-5.83	5.56e-09	0.185	-5.74	9.43e-09
b_time	   -1.49	0.186	-8.04	8.88e-16	0.181	-8.22	2.22e-16
sd_emipp	-1.02	0.142	-7.19	6.25e-13	0.143	-7.15	8.74e-13
sd_meals	-0.814	0.244	-3.33	0.000861	0.233	-3.5	0.000471
sd_petfr	1.3	0.328	3.96	7.56e-05	0.33	3.94	8.09e-05
u_emipp	   -2.08	0.242	-8.58	0	0.243	-8.55	0
u_meals  	1.7	0.216	7.89	2.89e-15	0.216	7.89	3.11e-15
u_petfr 	3.99	0.416	9.59	0	0.404	9.86	0

"""