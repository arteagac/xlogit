args = commandArgs(trailingOnly=TRUE)
if(length(args)== 2){
  n_draws = strtoi(args[1])
  n_cores = strtoi(args[2])
}else{
  n_draws = 1500
  n_cores = 4
}


### Load Apollo library
suppressMessages(library(apollo))

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="Apollo_MixedLogit_Swissmetro",
  modelDescr ="Mixed logit model on Swissmetro data",
  indivID   ="ID",  
  mixing    = TRUE, 
  nCores    = n_cores
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

df = read.table("http://transp-or.epfl.ch/data/swissmetro.dat", header=TRUE, sep="\t")
df = df[((df$PURPOSE == 1) | (df$PURPOSE == 3)) & (df$CHOICE!=0), ]
df$custom_id = 1:nrow(df)

#Pre-process columns
df$SM_CO = df$SM_CO * (df$GA == 0)
df$TRAIN_CO = df$TRAIN_CO * (df$GA == 0)

df$TRAIN_TT  = df$TRAIN_TT / 100.0
df$SM_TT = df$SM_TT / 100.0
df$CAR_TT = df$CAR_TT / 100
df$SM_CO = df$SM_CO /100
df$TRAIN_CO = df$TRAIN_CO /100
df$CAR_CO = df$CAR_CO /100


database = df
# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(
                ASC_CAR = 0,
                ASC_TRAIN = 0,
                B_CO = 0,
                u_TT = 0,
                sd_TT = 0.1
)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  intraDrawsType = "halton",
  intraNDraws    = 0,
  intraUnifDraws = c(),
  intraNormDraws = c(),
  interDrawsType = "halton",
  interNDraws    = n_draws,
  interUnifDraws = c(),
  interNormDraws = c("draws_TT")
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()
  randcoeff[["B_TT"]] = u_TT + sd_TT * draws_TT 
  return(randcoeff)
}

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs(silent = TRUE)

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
  V[['TRAIN']] = ASC_TRAIN + TRAIN_CO*B_CO + TRAIN_TT*B_TT
  V[['SM']] =               SM_CO*B_CO + SM_TT*B_TT
  V[['CAR']] =   ASC_CAR + CAR_CO*B_CO + CAR_TT*B_TT

  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(TRAIN=1, SM=2, CAR=3),
    avail         = list(TRAIN=TRAIN_AV, SM=SM_AV, CAR=CAR_AV),
    choiceVar     = CHOICE,
    V             = V
  )

  ### Compute probabilities using MNL model
  P[['model']] = apollo_mnl(mnl_settings, functionality)

  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)

  ### Average across inter-individual draws
  P = apollo_avgInterDraws(P, apollo_inputs, functionality)

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

apollo_modelOutput(model)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

log = function(msg){
    cat(msg,"\n", sep="")
    cat(msg, file="results/benchmark_results.out", sep="\n", append=TRUE)
}

#log(paste("draws=",n_draws," cores=", n_cores, " time(s)=", round(model$timeTaken, digits = 2),
#     " LogLik", round(model$LLout, digits = 2)))
#cat(paste("apollo", n_draws, n_cores, model$timeTaken, model$LLout, sep=","), 
#file = "results/benchmark_results_apollo_biogeme.csv", sep="\n", append=TRUE)

#LL(start)                        : -6922.01
#LL(0)                            : -6964.663
#LL(C)                            : -5864.998
#LL(final)                        : -4359.462

#Estimates:
#             Estimate        s.e.   t.rat.(0)    Rob.s.e. Rob.t.rat.(0)
#ASC_CAR        0.2803     0.05706       4.913      0.1086         2.582
#ASC_TRAIN     -0.5784     0.08332      -6.942      0.1488        -3.887
#B_CO          -1.6602     0.07792     -21.308      0.2918        -5.689
#u_TT          -3.2052     0.19601     -16.352      0.2369       -13.528
#sd_TT          3.6633     0.17864      20.507      0.2558        14.323

