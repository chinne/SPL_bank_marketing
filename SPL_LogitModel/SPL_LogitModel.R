balancedTrain <- read.csv("balancedTrain.csv", header = TRUE, sep = "," )
bankclean_test <- read.csv("bankclean_test.csv", header = TRUE, sep = ",")


##################################  Logit Regression   ################################

## Data Preparation: factor variables into dummy variables
install.packages("caret")
library(caret)
bank.train.dummy <- predict(dummyVars(y ~ . ,data=balancedTrain), newdata=balancedTrain) #use balanced train data
bank.train.dummy <- data.frame(bank.train.dummy, y=factor(balancedTrain$y))
bank.train.dummy$X<-NULL  #there is an extra variable created and should be deleted.
bank.test.dummy <- predict(dummyVars(y ~ . ,data=bankclean_test), newdata=bankclean_test)#use nonbalanced testdata
bank.test.dummy <- data.frame(bank.test.dummy, y=factor(bankclean_test$y))

## Training Logit modeling on train dataset
bank.train.dummy$y<-as.factor(ifelse(bank.train.dummy$y=="1","yes","no"))
logit <- train(y ~ ., data=bank.train.dummy, method="glm", 
               family = binomial("logit"), preProc = c("center", "scale"),metric="ROC",tuneLength=1,
               trControl=trainControl(method="cv", classProbs=TRUE,summaryFunction=twoClassSummary, 
                                      verboseIter=TRUE))

## model performance on train dataset
#confusion matrix
bank.train.class <- subset(bank.train.dummy, select=c(y), drop=TRUE)  #extract real behavour of the train data.
predict.logit.train1 <- predict.train(logit)  #use the model to make prediction on the train data and the output is class prediction
confusionMatrix(predict.logit.train1, bank.train.class, positive="yes")  #make a matrix to compare the prediciton and real situation.
# ROC and AUC
install.packages("PROC")
library(pROC)
predict.logit.train2<- predict(logit, type = "prob")[,2]  #extract prediction in probability format
logit.train.roc<-roc(bank.train.dummy$y,predict.logit.train2)#compare probability prediction with real behavior
auc(logit.train.roc)# calculate AUC based on ROC 
plot.roc(logit.train.roc)

## model performance on test dataset
#confusion matrix
bank.test.class <- subset(bank.test.dummy, select=c(y), drop=TRUE)
predict.logit.test1<-predict.train(logit,bank.test.dummy)#use the model to make prediction on the test data and the output is class prediction
confusionMatrix(predict.logit.test1, bank.test.class, positive="yes") 
#ROC and AUC
predict.logit.test2 <- predict(logit, bank.test.dummy, type="prob")[,2]#extract prediction in probability format
logit.test.roc<-roc(bank.test.dummy$y,predict.logit.test2)
auc(logit.test.roc)
plot.roc(logit.test.roc)
