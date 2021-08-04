# install library using: install.packages("mixl")
library(mixl)

args = commandArgs(trailingOnly=TRUE)
if(length(args)== 2){
  n_draws = strtoi(args[1])
  n_cores = strtoi(args[2])
}else{
  n_draws = 100
  n_cores = 4
}


df = read.csv("../data/artificial_wide.csv",header=TRUE)

df$ID = df$id
df$CHOICE = as.numeric(df$choice)


mnl_test <- "
b_emipp = @u_emipp + @sd_emipp * draw_emipp;
b_meals = @u_meals + @sd_meals * draw_meals;
b_petfr = @u_petfr + @sd_petfr * draw_petfr;

U_1 = $price_1*@b_price+$time_1*@b_time+$conven_1*@b_conven+$comfort_1*@b_comfort+$meals_1*b_meals+$petfr_1*b_petfr+$emipp_1*b_emipp+$nonsig1_1*@b_nonsig1+$nonsig2_1*@b_nonsig2+$nonsig3_1*@b_nonsig3;
U_2 = $price_2*@b_price+$time_2*@b_time+$conven_2*@b_conven+$comfort_2*@b_comfort+$meals_2*b_meals+$petfr_2*b_petfr+$emipp_2*b_emipp+$nonsig1_2*@b_nonsig1+$nonsig2_2*@b_nonsig2+$nonsig3_2*@b_nonsig3;
U_3 = $price_3*@b_price+$time_3*@b_time+$conven_3*@b_conven+$comfort_3*@b_comfort+$meals_3*b_meals+$petfr_3*b_petfr+$emipp_3*b_emipp+$nonsig1_3*@b_nonsig1+$nonsig2_3*@b_nonsig2+$nonsig3_3*@b_nonsig3;
"
est <- c(b_price = 0,b_time = 0,b_comfort = 0,b_conven = 0,b_nonsig1 = 0,b_nonsig2 = 0,b_nonsig3 = 0,u_emipp = 0,u_meals = 0,u_petfr = 0,sd_emipp = 0,sd_meals = 0,sd_petfr = 0)

model_spec <- mixl::specify_model(mnl_test, df, disable_multicore=F)

capture.output(
  model <- mixl::estimate(model_spec, start_values=est, data=df, nDraws=n_draws, num_threads=n_cores,
    availabilities = mixl::generate_default_availabilities(df, model_spec$num_utility_functions)), file='NULL')


log = function(msg){
    cat(msg,"\n", sep="")
    cat(msg, file="results/benchmark_results.out", sep="\n", append=TRUE)
}
tot_runtime= as.numeric(model$runtime, units="secs")
log(paste("draws=",n_draws," cores=", n_cores, " time(s)=", round(tot_runtime, digits = 2)," LogLik", round(model$maximum, digits = 2)))
cat(paste("mixl", n_draws, n_cores, tot_runtime, model$maximum, sep=","), file = "results/benchmark_results_apollo_biogeme.csv", sep="\n", append=TRUE)
#cat(paste("mixl", n_draws, n_cores, tot_runtime, model$maximum, sep=","), sep="\n", append=TRUE)  