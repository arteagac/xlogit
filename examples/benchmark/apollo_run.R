# This code estimates a mixed logit model for the artificial
# dataset using apollo. To use this code you must first 
# install apollo by executing:
# install.packages("apollo")

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="Apollo_MixedLogit_Artificial",
  modelDescr ="Mixed logit model on Artificial data",
  indivID   ="id",  
  mixing    = TRUE, 
  nCores    = 6
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

database = read.csv("../data/artificial_wide.csv",header=TRUE)

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(
                b_price = 0,
                b_time = 0,
                b_comfort = 0,
                b_conven = 0,
                b_nonsig1 = 0,
                b_nonsig2 = 0,
                b_nonsig3 = 0,
                u_emipp = 0,
                u_meals = 0,
                u_petfr = 0,
                sd_emipp = 0,
                sd_meals = 0,
                sd_petfr = 0
)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  interDrawsType = "",
  interNDraws    = 0,
  interUnifDraws = c(),
  interNormDraws = c(),
  intraDrawsType = "halton",
  intraNDraws    = 500,
  intraUnifDraws = c(),
  intraNormDraws = c("draws_emipp","draws_meals","draws_petfr")
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()
  randcoeff[["b_emipp"]] = u_emipp + sd_emipp * draws_emipp 
  randcoeff[["b_meals"]] = u_meals + sd_meals * draws_meals
  randcoeff[["b_petfr"]] = u_petfr + sd_petfr * draws_petfr
  return(randcoeff)
}

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){

  ### Function initialisation: do not change the following three commands
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  ### Create list of probabilities P
  P = list()

  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[['alt1']] = price_1*b_price+time_1*b_time+conven_1*b_conven+comfort_1*b_comfort+
    meals_1*b_meals+petfr_1*b_petfr+emipp_1*b_emipp+nonsig1_1*b_nonsig1+
        nonsig2_1*b_nonsig2+nonsig3_1*b_nonsig3

  V[['alt2']] = price_2*b_price+time_2*b_time+conven_2*b_conven+comfort_2*b_comfort+
      meals_2*b_meals+petfr_2*b_petfr+emipp_2*b_emipp+nonsig1_2*b_nonsig1+
          nonsig2_2*b_nonsig2+nonsig3_2*b_nonsig3

  V[['alt3']] = price_3*b_price+time_3*b_time+conven_3*b_conven+comfort_3*b_comfort+
      meals_3*b_meals+petfr_3*b_petfr+emipp_3*b_emipp+nonsig1_3*b_nonsig1+
          nonsig2_3*b_nonsig2+nonsig3_3*b_nonsig3

  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(alt1=1, alt2=2, alt3=3),
    avail         = list(alt1=1, alt2=1, alt3=1),
    choiceVar     = choice,
    V             = V
  )

  ### Compute probabilities using MNL model
  P[['model']] = apollo_mnl(mnl_settings, functionality)

  ### Take product across observation for same individual
  #P = apollo_panelProd(P, apollo_inputs, functionality)

  ### Average across inter-individual draws
  P = apollo_avgIntraDraws(P, apollo_inputs, functionality)

  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed,
                        apollo_probabilities, apollo_inputs, 
                        estimate_settings=list(hessianRoutine="maxLik"))

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)

# =============== OUTPUT OF THE ESTIMATION ================
#Model run using Apollo for R, version 0.1.0 
#www.ApolloChoiceModelling.com

#Model name                       : Apollo_MixedLogit_Artificial
#Model description                : Mixed logit model on Artificial data
#Model run at                     : 2020-10-22 01:32:00
#Estimation method                : bfgs
#Model diagnosis                  : successful convergence 
#Number of individuals            : 4000
#Number of observations           : 4000
#
#Number of cores used             :  6 
#Number of intra-person draws     : 500 (halton)
#
#LL(start)                        : -4394.449
#LL(0)                            : -4394.449
#LL(final)                        : -2278.603
#Rho-square (0)                   :  0.4815 
#Adj.Rho-square (0)               :  0.4785 
#AIC                              :  4583.21 
#BIC                              :  4665.03 
#Estimated parameters             :  13
#Time taken (hh:mm:ss)            :  00:07:17.49 
#Iterations                       :  52 
#Min abs eigenvalue of hessian    :  3.216601 
#
#Estimates:
#          Estimate Std.err. t.ratio(0) Rob.std.err. Rob.t.ratio(0)
#b_price    -1.0359   0.1741      -5.95       0.1762          -5.88
#b_time     -1.4631   0.1762      -8.30       0.1725          -8.48
#b_comfort   1.0662   0.1896       5.62       0.1850           5.76
#b_conven    0.8962   0.1534       5.84       0.1518           5.90
#b_nonsig1   0.0747   0.1461       0.51       0.1441           0.52
#b_nonsig2   0.0140   0.1532       0.09       0.1477           0.09
#b_nonsig3   0.0174   0.1332       0.13       0.1296           0.13
#u_emipp    -2.0209   0.2239      -9.03       0.2176          -9.29
#u_meals     1.7115   0.1992       8.59       0.1931           8.86
#u_petfr     3.8765   0.3773      10.27       0.3486          11.12
#sd_emipp    1.0199   0.1396       7.30       0.1435           7.11
#sd_meals   -0.6946   0.2565      -2.71       0.2556          -2.72
#sd_petfr    1.2563   0.3130       4.01       0.3028           4.15
#
#Overview of choices for model component "MNL"
#                                    alt1    alt2    alt3
#Times available                  4000.00 4000.00 4000.00
#Times chosen                      237.00 3001.00  762.00
#Percentage chosen overall           5.92   75.02   19.05
#Percentage chosen when available    5.92   75.02   19.05

