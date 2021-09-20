options(repos="http://cran.us.r-project.org")

install.packages("mlogit")
install.packages("apollo")
install.packages("gmnl")
install.packages("mixl")

# Below is the ideal way to install the R packages to keep
# consistency in the versions. Unfortunately, in MS Windows
# this requires to install compilers and R tools. For 
# this reason, the default versions available in CRAN are used.

#install.packages("devtools")
#require("devtools")
#install_version("apollo", version = "0.1.0")
#install_version("mlogit", version = "1.1-1")