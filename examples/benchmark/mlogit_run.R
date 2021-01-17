#install.packages("mlogit")
suppressMessages(library(mlogit))
data_folder = "../data/"
args = commandArgs(trailingOnly=TRUE)
if(length(args)== 4){
    n_draws = strtoi(args[1])
    dataset = args[2]
    profile = args[4] == 'prof'
}else{ # Args by default
    n_draws = 100
    dataset = 'artificial'
    profile = FALSE
}

log = function(msg){
    cat(msg,"\n", sep="")
    cat(msg,"\n", file="results/benchmark_results.out", sep="", append=TRUE)
}

if(dataset == "artificial"){
    df = read.csv(paste(data_folder, "artificial_long.csv", sep = ""))
    Artif <- mlogit.data(df, shape="long", id.var="id", choice="choice", alt.var="alt") 
    run_estimation = function(){
        mlogit(choice~price+time+conven+comfort+meals+petfr+emipp+nonsig1+nonsig2+nonsig3|0, Artif,
            rpar=c(meals="n", petfr="n", emipp="n"), 
            R=n_draws,halton=NA,print.level=0)
    }
}
if(dataset == "electricity"){
    df = read.csv(paste(data_folder, "electricity_long.csv", sep = ""))
    Electr <- mlogit.data(df, shape="long", id.var="id", chid.var="chid", choice="choice", alt.var="alt")
    run_estimation = function(){
        model = mlogit(choice~pf+cl+loc+wk+tod+seas|0, Electr,
                rpar=c(pf="n",cl="n",loc="n",wk="n",tod="n",seas="n"), 
                R=n_draws,halton=NA,print.level=0,panel=TRUE)
    }
}


if(profile){
    Rprof(interval = 0.05, memory.profiling = TRUE)
    model = run_estimation()
    Rprof(append=TRUE)
    time = (summaryRprof("Rprof.out"))$by.total[1, "total.time"]
    sm = summaryRprof("Rprof.out", memory = "tseries", diff = FALSE)
    sm$tot = sm$vsize.small + sm$vsize.large + sm$nodes
    mem = max(sm$tot)/(1024*1024*1024)
    #Format and print output
    res = c()
    res = c(res, format(n_draws, width = 6, justify = "right", trim=T))
    res = c(res, format(time, width = 7, nsmall = 2, digits = 2, justify = "right", trim=T))
    res = c(res, format(model$logLik[1], width = 11, digits = 2, nsmall = 2, justify = "right", trim=T))
    res = c(res, format(mem, width = 7, digits = 3, justify = "right", trim=T))
    res = c(res, format(0, width = 7, digits = 3, justify = "right", trim=T))
    res = c(res, format(model$est.stat$code == 1, width = 5, justify = "left", trim=T))
    log(paste(res, collapse=" "))
    cat(paste("mlogit", dataset, n_draws, time, model$logLik[1], mem, 0,
        model$est.stat$code == 1, sep=","), file = "results/benchmark_results.csv", sep="\n", append=TRUE)
    #invisible(file.remove("Rprof.out"))
}else{
    model = run_estimation()
    summ = coef(summary(model))
    varnames = row.names(summ)
    est_coeff = summ[,"Estimate"]
    est_stder = summ[,"Std. Error"]
    log("Variable    Estimate   Std.Err.")
    for (i in 1:nrow(summ)){
        log(paste(c(format(varnames[i], width=10, trim=T),
            format(est_coeff[i], width=10, nsmall = 5, digits = 5, trim=T),
                format(est_stder[i], width=9, nsmall = 5, digits = 5, trim=T)),
                collapse=""))
    }
    log(paste("Log.Lik:   ", format(logLik(model)[1], nsmall = 2, digits = 2, trim=T)))
}



