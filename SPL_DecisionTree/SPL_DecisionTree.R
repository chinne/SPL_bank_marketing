# import libraries
library(caret)
library(rpart)
library(rpart.plot)
library(lattice)
library(ggplot2)
library(e1071)
library(psych)
library(plyr)
library(ROCR)
library(AUC)
library(party)
library(partykit)
library(xlsx)
library(mlr)

# read data
bank = read.csv("bankclean.csv")
bank = bank[, -1]


# split the data for training and testing

set.seed(124)
n = nrow(bank)
sample.size = ceiling(n * 0.8)
idx.train = sample(n, sample.size)

data_train = bank[idx.train, ]
data_test = bank[-idx.train, ]

# load the balanced training set and

balancedTrain = read.csv("balancedTrain.csv")

levels(balancedTrain$default) = levels(bank$default)


# build the decison tree model

# pre-pruning the tree in case of overfitting
rpart.control = rpart.control(minsplit = 5, minbucket = round(5/3), maxdepth = 4, 
    cp = 0.001)
# decision tree model
dtm = rpart(y ~ ., data = balancedTrain, method = "class", parms = list(split = "information"), 
    control = rpart.control)

# plot the decision tree model
rpart.plot(dtm, type = 1, cex = 0.6, extra = 104, tweak = 0.8, faclen = 3)

# test the classifier in testing dataset

pred_dtm = predict(dtm, newdata = data_test, type = "prob")

# set the threshold to see the Confusion Matrix
threshold = 0.5
pred_class = as.matrix(factor(ifelse(pred_dtm[, 2] > threshold, "yes", 
    "no")))
confusionMatrix(pred_class, data_test$y, positive = "yes")



# plot the ROC and calculate AUC


roc_dtm = roc(pred_dtm[, 2], data_test$y)
roc_dtm

plot(roc_dtm, main = "ROC Curve for Decision Tree Model", col = "red")

auc_dt_b = auc(roc_dtm, min = 0, max = 1)
auc_dt_b



