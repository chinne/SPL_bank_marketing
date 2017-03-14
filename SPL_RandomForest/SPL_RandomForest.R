train <- read.csv("balancedTrain.csv", header = TRUE, sep = ",", )
bankclean <- read.csv("bankclean.csv", header = TRUE, sep = ",")
bankclean$X <- NULL

#################################   splitting data   ###############################
set.seed(124)
n <- nrow(bankclean) 
sample.size <- ceiling(n*0.8) 
idx.train <- sample(n, sample.size) 
bank_test <-  bankclean[-idx.train, ]

train$X <- NULL
train$y <- ifelse(train$y == 1, "yes", "no")
train$y <- as.factor(train$y)


library(mlr)
library(h2o)
set.seed(1234)
bank.train <- makeClassifTask(data = train,target="y", positive = "yes")  #Creating a Classification Task based on the balanced train data set
bank.test <- makeClassifTask(data = bank_test,target="y", positive = "yes")  #Creating a Classification Task based on the balanced train data se
tune.ctrl <- makeTuneControlGrid(tune.threshold = TRUE)
cvDesc <- makeResampleDesc(method = "CV", iters = 5, stratify = T)
rf.lrn = makeLearner("classif.h2o.randomForest",
                     predict.type = "prob",
                     fix.factors.prediction = TRUE)

rf.PS = makeParamSet(
  makeDiscreteParam("ntrees", values = seq(500,1000,500)),
  makeDiscreteParam("mtries", values = seq(1,10,1))
)

h2o.init(nthreads = -1)

rf.para <- tuneParams(rf.lrn, task = bank.train, resampling = cvDesc, par.set = rf.PS, control = tune.ctrl, measures = auc)

h2o.shutdown()


library(kernlab)
lrn = setHyperPars(rf.lrn, par.vals = rf.para$x)
rf.mod <- mlr::train(lrn,task = bank.train)

yhat.rf <- predict(rf.mod, task = bank.train)
predict.rf <- predict(rf.mod, task = bank.test)
#####Performance Measures################


confMatrix_train <- calculateConfusionMatrix(yhat.rf, relative = TRUE)
confMatrix_test <- calculateConfusionMatrix(predict.rf, relative = TRUE)

performance(yhat.rf, measures = auc)
performance(predict.rf, measures = auc)

plotLearnerPrediction(rf.lrn, task = bank.test)

rf.plotObj <- generateThreshVsPerfData(predict.rf, measures = list(fpr,tpr))
rf.auc.plot <- plotROCCurves(rf.plotObj, measures = list(fpr,tpr))
rf.auc.plot