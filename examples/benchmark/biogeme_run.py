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
import sys
from time import time
from tools import log as plog

data_folder = "../data/"

if len(sys.argv) == 3:  # If CLI arguments provided
    n_draws, n_cores = int(sys.argv[1]), int(sys.argv[2])
else:  # Default in case CLI arguments not provided
    n_draws, n_cores = 500, 2


df = pd.read_csv(data_folder+"artificial_wide.csv")
df['choice'] = df['choice'].astype('str')
mapping = {'1': 1, '2': 2, '3': 3}

for k, v in mapping.items():
    df["aval_"+k] = np.ones(df.shape[0])
start_time = time()
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
logger.setSilent()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=n_draws,
                      numberOfThreads=n_cores)
biogeme.modelName = 'MixedLogitArtificial'
biogeme.generateHtml = False
biogeme.generatePickle = False
# Estimate the parameters
results = biogeme.estimate()
ellapsed = time() - start_time

plog("draws={} cores={} time(s)={:.2f} LogLik={:.2f}"
     .format(n_draws, n_cores, ellapsed, results.data.logLike))

with open("results/benchmark_results_apollo_biogeme.csv", 'a') as fw:
    fw.write("{},{},{},{},{}\n".format("biogeme", n_draws, n_cores,
                                       ellapsed, results.data.logLike))
