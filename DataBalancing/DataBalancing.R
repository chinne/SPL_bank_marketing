# import libraries
library(unbalanced)
library(FNN)

# adjust the response variable to 0/1 factor

data_train$y=as.factor(ifelse(data_train$y=="yes","1","0"))

summary(data_train$y)

# calculate the SMOTE resamping scale parameters
sum = length(data_train$y);sum
ori_min = length(which(data_train$y == 1));ori_min
new_min = round(sum*(1/2)) - ori_min ;new_min
over_s = (round(sum*(1/2)) / ori_min) - 1 ;over_s
un_s = (sum - all_min)/new_min ; un_s

# implement SMOTE
newdata = ubSMOTE(data_train[,-20],data_train[,20],
                  k = 10,
                  perc.over = 100*over_s,
                  perc.under = 100*un_s,
                  verbose = FALSE)

#put the newdata to the same dataframe as original training set
balancedTrain = as.data.frame(newdata)
colnames(balancedTrain) = colnames(data_train)

#round the value for variables whose value requires integer
balancedTrain$age = round(balancedTrain$age)
balancedTrain$campaign = round(balancedTrain$campaign)
balancedTrain$previous = round(balancedTrain$previous)

summary(balancedTrain$y)

write.csv(balancedTrain, "balancedTrain.csv")