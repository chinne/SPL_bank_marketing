rm(list=ls())

if(!require("nnet")) install.packages("nnet"); library("nnet") 
if(!require("pROC")) install.packages("pROC"); library("pROC") 
if(!require("caret")) install.packages("caret"); library("caret") 
if(!require("devtools")) install.packages("devtools"); library("devtools")
if(!require("reshape")) install.packages("reshape"); library("reshape")
if(!require("hmeasure")) install.packages("hmeasure"); library("hmeasure")


# Read data
df <- read.csv("balancedTrain.csv",
               sep = ",")

# Delete rownumber
df$X  <- NULL
# Delete default variable
df$default <- NULL

# Save outcome variable as Yes/No
df$y <- as.factor(df$y)
df$y <- ifelse(df$y==1, "Yes", "No")




###################   splitting data   ######################
set.seed(124)
n <- nrow(df) 
sample.size <- ceiling(n*0.8) # Set training sample size to 80 pct. of data
idx.train <- sample(n, sample.size) # Draw a random, stratified sample including p percent of the data
train <- df[idx.train, ] # training set
test <-  df[-idx.train, ] # test set (drop all observations with train indeces)

model.control<- trainControl(
  method = "cv", # 'cv' for cross validation
  number = 5, # number of folds in cross validation
  classProbs = TRUE, # Calculate class probabilities in each resample
  summaryFunction = twoClassSummary, # Compute sensitivity, specificity and AUC for each resample
  returnData = FALSE # The training data will not be included in the ouput training object
)

# Define a search grid of values to test
nn.parms <- expand.grid(decay = c(0, 10^seq(-3, 0, 1)), size = seq(3,15,1))

#Train neural network nn with 5-fold cv
nn <- train(y~., data = train,  
            method = "nnet", 
            maxit = 200, # specify max number of iterations
            trace = FALSE, # options for nnet function
            tuneGrid = nn.parms, # parameters to be tested
            metric = "ROC", trControl = model.control)
# Analyze the cross-validation results
print(nn)
plot(nn)

# Query the internal structure of the network 
summary(nn)
# Plot the neural network
plot.nnet(nn)

# We can now predict the outcomes of the test set and assess these in terms of AUC or other performance criteria.
yhat.nn   <- predict(nn, newdata = test, type = "prob")[,2]
# Obtain ROC curve and calculate AUC 
nn.roc <-roc(test$y, yhat.nn)

# Calculate auc and plot roc graph.
auc(nn.roc)
plot.roc(nn.roc)

#Calculate confusion matrix with threshold of 0.5
test$prediction <- yhat.nn
test$classification <- ifelse(test$prediction >=0.5, 1,0)
test$y <- ifelse(test$y=="Yes", 1, 0)

truepositive <- sum(test$y==test$classification & test$y==1)
truenegative <- sum(test$y==test$classification & test$y==0)
falsepositive <- sum(test$classification==1 & test$y==0)
falsenegative <- sum(test$classification==0 & test$y==1)

