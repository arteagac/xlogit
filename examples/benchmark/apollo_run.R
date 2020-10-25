# This code estimates a mixed logit model for the artificial
# dataset using apollo. For installation:
# Install R version 4.0.3 and then run
# install.packages("https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.9.850.1.0.tar.gz",repos=NULL,type="source")
# install.packages(c("maxLik", "mnormt", "mvtnorm", "coda", "sandwich", "randtoolbox", "numDeriv", "RSGHB", "Deriv"))
# install.packages("http://www.apollochoicemodelling.com/files/apollo_0.1.0.tar.gz",repos=NULL,type="source")
# These specific versions need to be kept. Otherwise Apollo won't install

args = commandArgs(trailingOnly=TRUE)
if(length(args)== 2){
  n_draws = strtoi(args[1])
  n_cores = strtoi(args[2])
}else{
  n_draws = 500
  n_cores = 6
}


### Load Apollo library
suppressMessages(library(apollo))

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="Apollo_MixedLogit_Artificial",
  modelDescr ="Mixed logit model on Artificial data",
  indivID   ="id",  
  mixing    = TRUE, 
  nCores    = n_cores
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
  intraNDraws    = n_draws,
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

apollo_inputs = apollo_validateInputs(silent=TRUE)

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
estimate_settings = list(silent=TRUE, writeIter=FALSE)
#invisible(capture.output(
  model <- apollo_estimate(apollo_beta, apollo_fixed,
                        apollo_probabilities, apollo_inputs, 
                        estimate_settings=estimate_settings)
#))

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

cat(c("draws=",n_draws," cores=", n_cores, " time(s)=", round(model$timeTaken, digits = 2),
     " LogLik", round(model$LLout, digits = 2)), sep="")
cat("\n")
cat(paste("apollo", n_draws, n_cores, model$timeTaken, model$LLout, sep=","), 
file = "results/benchmark_results_apollo_biogeme.csv", sep="\n", append=TRUE)
