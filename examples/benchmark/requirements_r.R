options(repos="http://cran.us.r-project.org")
install.packages("mlogit")
install.packages("apollo")

# These were used before but it throws errors when a proper c compiler is not available.
#install.packages("devtools")
#require(devtools)
#install_version("mlogit", version = "1.1-1")
#install_version("apollo", version = "0.1.0")