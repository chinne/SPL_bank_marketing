setwd("C:/Users/yufangi/Documents/SPL/Project")
bank <- read.csv("bank-additional-full.csv", header = TRUE, sep = ";")

##################################   Preparing dataset  ##############################
##  removing variable duration and default
bank$duration <-NULL
bank$default <-NULL

##  changing variable pdays
bank$pdays <- ifelse(bank$pdays == 999, 0, 1)

## the total number of missing value
NAs <- bank == "unknown"
is.na(bank)[NAs] <- TRUE

################################### Data Imputation #######################
## ploting missing value pattern
install.packages("VIM")
library(VIM)
mice_plot<-aggr(bank,col=c('navyblue','yellow'),
                numbers=TRUE,sortVars=TRUE,labels=names(bank),
                cex.axis=.7,gap=3,ylab=c("Missing_Data_Ratio","Missing_Data_Pattern"))

## splitting data into train and test set  
set.seed(124)
n <- nrow(bank) 
sample.size <- ceiling(n*0.8) 
idx.train <- sample(n, sample.size) 
bank_train <- bank[idx.train, ] 
bank_test <-  bank[-idx.train, ]

## imputing data with mice package
install.packages("mice")
library(mice)
# Data Imputing for Train Dataset
tempData1 <- mice(bank_train,m=5,maxit=10,meth="polyreg",seed=500,diagnostics=True)
bankclean_train <- complete(tempData1,1)
write.csv(bankclean_train, "bankclean_train.csv")
# Data Imputing for Test Dataset
tempData2 <- mice(bank_test,m=5,maxit=10,meth="polyreg",seed=500,diagnostics=True)
pred <- tempData2$predictorMatrix
pred[,"y"] <- 0
tempData3 <- mice(bank_test, pred=pred, pri=F)
bankclean_test <- complete(tempData3,1)
write.csv(bankclean_test, "bankclean_test.csv")
