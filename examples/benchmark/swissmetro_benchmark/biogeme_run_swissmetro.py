"""File 05normalMixture.py

:author: Michel Bierlaire, EPFL
:date: Sat Sep  7 18:23:01 2019

 Example of a mixture of logit models, using Monte-Carlo integration.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
from biogeme.expressions import Beta, DefineVariable, bioDraws, log, MonteCarlo
from biogeme.expressions import PanelLikelihoodTrajectory

# Read the data
df = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to investigate the database. For example:
#print(database.data.describe())

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)
database.panel("ID")
# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0), database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0), database)
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0, database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100, database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0, database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100, database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100, database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100, database)

# Definition of the utility functions
V1 = ASC_TRAIN + \
     B_TIME_RND * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME_RND * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME_RND * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}
 
# Conditional to B_TIME_RND, we have a logit model (called the kernel)
prob = models.logit(V, av, CHOICE)

# Add Panel Structure
pnl_prob = PanelLikelihoodTrajectory(prob)
# We integrate over B_TIME_RND using Monte-Carlo
logprob = log(MonteCarlo(pnl_prob))

# Define level of verbosity
logger = msg.bioMessage()
#logger.setSilent()
#logger.setWarning()
logger.setGeneral()
#logger.setDetailed()

# These notes will be included as such in the report file.
userNotes = ('Example of a mixture of logit models with three alternatives, '
             'approximated using Monte-Carlo integration.')

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=1500, userNotes=userNotes)
biogeme.modelName = '05normalMixture'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)

"""
OUTPUT: 
Log likelihood (N = 752):   -4360.56 Gradient norm:       0.02 Hessian norm:       7e+02 BHHH norm:       4e+03
              Value   Std err     t-test       p-value  Rob. Std err  Rob. t-test  Rob. p-value
ASC_CAR    0.277854  0.055766   4.982508  6.276531e-07      0.105379     2.636708  8.371494e-03
ASC_TRAIN -0.585135  0.078689  -7.436023  1.036948e-13      0.138438    -4.226694  2.371499e-05
B_COST    -1.655086  0.077745 -21.288720  0.000000e+00      0.293532    -5.638518  1.715196e-08
B_TIME    -3.178309  0.168664 -18.843978  0.000000e+00      0.195959   -16.219249  0.000000e+00
B_TIME_S   3.678093  0.169751  21.667619  0.000000e+00      0.231974    15.855631  0.000000e+00
"""