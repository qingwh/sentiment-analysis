
# Title: Sentiment analysis for Galaxy in R

# Last update: 2018.10.20

# File:  sentiment analysis.R

###############
# Project Notes
###############

# Summarize project:  We will develop several models and pick the best model using Small matrix dataset which are manually labeled with a sentiment rating for Galaxy.
# We will then use the best model to predict sentiment ratings of the large matrix dataset for Galaxy.

################
# Load packages
################
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("doParallel")
install.packages('e1071', dependencies=TRUE)
install.packages("C50")
install.packages("kknn")
install.packages("gbm")
library(caret)
library(corrplot)
library(readr)
library(e1071)
library(C50)
library(kknn)
library(gbm)

#####################
# Parallel processing
#####################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-2, type='PSOCK')
registerDoParallel(cl)

###############
# Import data
##############

## Load training and test set
galaxy_smallmatrix<- read.csv("C:/Users/admin/galaxy_smallmatrix_labeled_9d.csv", stringsAsFactors = FALSE, header=T)
galaxy_largematrix<- read.csv("C:/Users/admin/galaxyLargeMatrix.csv", stringsAsFactors = FALSE, header=T)

################
# Evaluate data
################

#--- Training Set ---#
summary(galaxy_smallmatrix) 
str(galaxy_smallmatrix) 
head(galaxy_smallmatrix)
tail(galaxy_smallmatrix)
table(galaxy_smallmatrix$galaxysentiment)

# check for missing values 
is.na(galaxy_smallmatrix)
any(is.na(galaxy_smallmatrix))

# plot
hist(galaxy_smallmatrix$galaxysentiment)
qqnorm(galaxy_smallmatrix$galaxysentiment)  

#--- Prediction Set ---# 
summary(galaxy_largematrix) 
str(galaxy_largematrix) 
head(galaxy_largematrix)
tail(galaxy_largematrix)

# check for missing values 
is.na(galaxy_largematrix)
any(is.na(galaxy_largematrix))

#############
# Preprocess
#############

#--- Training and test set ---#

# change variable types
galaxy_smallmatrix$galaxysentiment <- as.factor(galaxy_smallmatrix$galaxysentiment)
# normalize
preprocessParamsg <- preProcess(galaxy_smallmatrix[,1:58], method = c("center", "scale"))
print(preprocessParamsg)
galaxy_smallmatrix_N <- predict(preprocessParams, galaxy_smallmatrix)
str(galaxy_smallmatrix_N)

#--- Prediction set ---#

# drop ID variable
galaxy_largematrix$id<- NULL
# change variable types
galaxy_largematrix$galaxysentiment <- as.factor(galaxy_largematrix$galaxysentiment)
# normalize
galaxy_largematrix_N <- predict(preprocessParamsg, galaxy_largematrix)
str(galaxy_largematrix_N)

##################
# Train/test sets
##################

# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTrainingg <- createDataPartition(galaxy_smallmatrix_N$galaxysentiment, p=0.7, list=FALSE)
# create training/testing dataset
gtrainSetN <- galaxy_smallmatrix_N[inTrainingg,]   
gtestSetN <- galaxy_smallmatrix_N[-inTrainingg,]   

# verify number of obs 
nrow(gtrainSetN)
nrow(gtestSetN)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

##############
# Train model
##############
## ------- Decision Tree C5.0 ------- ##
# set random seed
set.seed(123)
# train/fit
C5.0_Fitg <- train(galaxysentiment~., data=gtrainSetN, method="C5.0", trControl=fitControl,metric = "Kappa", tuneLength=5)
C5.0_Fitg

#C5.0 

#9040 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:

#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7635838  0.5257363
#rules  FALSE   10      0.7576107  0.5158336
#rules  FALSE   20      0.7576107  0.5158336
#rules  FALSE   30      0.7576107  0.5158336
#rules  FALSE   40      0.7576107  0.5158336
#rules   TRUE    1      0.7625660  0.5234605
#rules   TRUE   10      0.7556638  0.5103959
#rules   TRUE   20      0.7556638  0.5103959
#rules   TRUE   30      0.7556638  0.5103959
#rules   TRUE   40      0.7556638  0.5103959
#tree   FALSE    1      0.7631191  0.5259074
#tree   FALSE   10      0.7569688  0.5153591
#tree   FALSE   20      0.7569688  0.5153591
#tree   FALSE   30      0.7569688  0.5153591
#tree   FALSE   40      0.7569688  0.5153591
#tree    TRUE    1      0.7619687  0.5235390
#tree    TRUE   10      0.7548891  0.5097109
#tree    TRUE   20      0.7548891  0.5097109
#tree    TRUE   30      0.7548891  0.5097109
#tree    TRUE   40      0.7548891  0.5097109

#Kappa was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = tree and winnow = FALSE.

#--- Save top performing model ---#

saveRDS(C5.0_Fitg, "C5.0_Fitg.rds")  
# load and name model
C5.0_Fitg <- readRDS("C5.0_Fitg.rds")

#---Predict testSet with  C5.0---#

# predict with C5.0
C5.0_Predg <- predict(C5.0_Fitg, gtestSetN)
C5.0_Predg 
#summarize predictions
summary(C5.0_Predg)

# performance measurement
postResample(C5.0_Predg, gtestSetN$galaxysentiment)

#Accuracy     Kappa 
#0.7675019 0.5315174 
# plot
plot(C5.0_Predg,gtestSetN$galaxysentiment)

#calculate confusion Matrix
options(max.print=10000)
gcmC5.0 <-confusionMatrix(C5.0_Predg,gtestSetN$galaxysentiment, mode="everything")
gcmC5.0 

## ------- random forest ------- ##

# set random seed
set.seed(123)
# train/fit
rf_Fitg <- train(galaxysentiment~., data=gtrainSetN, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_Fitg

#Random Forest 

#9040 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:

#  mtry  Accuracy   Kappa    
#2    0.7039816  0.3534120
#16    0.7670127  0.5338944
#30    0.7633183  0.5300115
#44    0.7592696  0.5241720
#58    0.7549995  0.5175906

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 16.

#--- Save top performing model ---#
saveRDS(rf_Fitg, "rf_Fitg.rds")  
# load and name model
rf_Fitg <- readRDS("rf_Fitg.rds")

#---Predict testSet with  rf---#
# predict with rf
rf_Predg <- predict(rf_Fitg, gtestSetN)
rf_Predg 
#summarize predictions
summary(rf_Predg)
# performance measurement
postResample(rf_Predg, gtestSetN$galaxysentiment)
#Accuracy     Kappa 
#0.7693103 0.5371969 

#calculate confusion Matrix
cmrfg <-confusionMatrix(rf_Predg,gtestSetN$galaxysentiment, mode="everything")
cmrfg

## ------- SVM ------- ##

# set random seed
set.seed(123)
# SVM train/fit
svm_Fitg <- train(galaxysentiment~., data=gtrainSetN, method="svmLinear", trControl=fitControl, metric = "Kappa",tuneLength=5) 
svm_Fitg

# Support Vector Machines with Linear Kernel 

# 9040 samples
# 58 predictor
# 6 classes: '0', '1', '2', '3', '4', '5' 

# No pre-processing
# Resampling: Cross-Validated (5 fold, repeated 5 times) 
# Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
# Resampling results:

#   Accuracy   Kappa    
# 0.7121678  0.3953766

# Tuning parameter 'C' was held constant at a value of 1


#--- Save top performing model ---#

saveRDS(svm_Fitg, "svmFitg.rds")  
# load and name model
svm_Fitg<- readRDS("svmFitg.rds")

## ------- KKNN------- ##

# set random seed
set.seed(123)

# KKNN train/fit
kknn_Fitg <- train(galaxysentiment~., data=gtrainSetN, method="kknn", trControl=fitControl, metric = "Kappa",tuneLength=5) 
kknn_Fitg

#k-Nearest Neighbors 

#9040 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:

#  kmax  Accuracy   Kappa    
#5    0.6651349  0.4205785
#7    0.7190057  0.4754396
#9    0.7402220  0.4956944
#11    0.7508407  0.5076571
#13    0.7519685  0.5093890

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a
#value of optimal
#Kappa was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 13, distance = 2 and kernel = optimal.

#--- Save top performing model ---#

saveRDS(kknn_Fitg, "kknnFitg.rds")  
# load and name model
kknn_Fitg<- readRDS("kknnFitg.rds")

#---Predict testSet with kknn---#

# predict with kknn
kknnPredg <- predict(kknn_Fitg, gtestSetN)
# print predictions
kknnPredg
#summarize predictions
summary(kknnPredg)
#performance measurement
postResample(kknnPredg, gtestSetN$galaxysentiment)
#Accuracy     Kappa 
#0.7574270 0.5154767 

#calculate confusion Matrix
cmkknng <-confusionMatrix(kknnPredg1, gtestSetN$galaxysentiment)


## ------- GBM------- ##

#recode variable of training set
gtrainSetNG <-gtrainSetN
gtrainSetNG$galaxysentiment <- recode(gtrainSetN$galaxysentiment, '0' = 'seg0', '1' = 'seg1', '2' = 'seg2', '3' = 'seg3', '4' = 'seg4', '5' = 'seg5')
str(gtrainSetNG)

#recode variable of test set
gtestSetNG <-gtestSetN
gtestSetNG$galaxysentiment <- recode(gtestSetNG$galaxysentiment, '0' = 'seg0', '1' = 'seg1', '2' = 'seg2', '3' = 'seg3', '4' = 'seg4', '5' = 'seg5')
str(gtestSetNG)

# set random seed
set.seed(123)
# GBM train/fit
gmodel3<- train(galaxysentiment~., data=gtrainSetNG,method='gbm',trControl = fitControl,verbose=F,tuneLength=3)
gmodel3

#Stochastic Gradient Boosting 

#9040 samples
#58 predictor
#6 classes: 'seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:

#  interaction.depth  n.trees  Accuracy   Kappa    
#1                   50      0.7213942  0.4114028
#1                  100      0.7384290  0.4573261
#1                  150      0.7473007  0.4802339
#2                   50      0.7470132  0.4809381
#2                  100      0.7506857  0.4901796
#2                  150      0.7601326  0.5153534
#3                   50      0.7550000  0.5016542
##3                  100      0.7640705  0.5259049
#3                  150      0.7651324  0.5291339

#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning parameter 'n.minobsinnode' was held
#constant at a value of 10
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode
#= 10.

#--- Save top performing model ---#

saveRDS(gmodel3, "gbmFit3g.rds")  
# load and name model
gbm_Fit3g<- readRDS("gbmFit3g.rds")

#---Predict testSet---#

# predict with gbm
gbmPred1g <- predict(gbm_Fit3g, gtestSetNG)
# print predictions
gbmPred1g
# summarize predictions
summary(gbmPred1g)
# performance measurement
postResample(gbmPred1g, gtestSetNG$galaxysentiment)
#Accuracy     Kappa 
#0.7712082 0.557140

#calculate confusion Matrix
cmgbmg <-confusionMatrix(gbmPred1g, gtestSetNG$galaxysentiment, mode="everything")

##--- Compare metrics ---##

ModelFitResultsg <- resamples(list(C5.0=C5.0_Fitg,rf=rf_Fitg,SVM=svm_Fitg,kknn=kknn_Fitg,gbm=gbm_Fit3g))
# output summary metrics for tuned models 
summary(ModelFitResultsg)


#Call:
#  summary.resamples(object = ModelFitResultsg)

#Models: C5.0, rf, SVM, kknn, gbm 
#Number of resamples: 25 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C5.0 0.7483407 0.7599558 0.7628524 0.7631191 0.7665929 0.7809735    0
#rf   0.7553957 0.7631433 0.7667219 0.7670127 0.7693584 0.7826327    0
#SVM  0.6950747 0.7072496 0.7140487 0.7121678 0.7182320 0.7367257    0
#kknn 0.7293857 0.7470946 0.7538717 0.7519685 0.7588496 0.7672747    0
#gbm  0.7515219 0.7625899 0.7648035 0.7651324 0.7678275 0.7798673    0

#Kappa 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C5.0 0.4980952 0.5188530 0.5255550 0.5259074 0.5314367 0.5651669    0
#rf   0.5103113 0.5248984 0.5339376 0.5338944 0.5389400 0.5691963    0
#SVM  0.3521243 0.3867967 0.3970922 0.3953766 0.4084361 0.4532139    0
#kknn 0.4755206 0.4986195 0.5117041 0.5093890 0.5172126 0.5378172    0
#gbm  0.5009611 0.5237389 0.5286343 0.5291339 0.5362801 0.5608608    0


#######################################################
# Random Forest Model with Feature Engineering method 1
#######################################################

options(max.print=1000000)
galaxyCOR <- galaxy_smallmatrix_N
corrAll <- cor(galaxyCOR[,1:59])
corrAll 

# plot correlation matrix
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
gcorr58<- cor(galaxyCOR[,1:58])

# create object with indexes of highly corr features
gcorrIVhigh <- findCorrelation(gcorr58, cutoff=0.8)
# print indexes of highly correlated attributes
gcorrIVhigh

# get var name of high corr IV
colnames(galaxyCOR[c(29)]) #samsungdisneg
colnames(galaxyCOR[c(44)]) #samsungperneg
colnames(galaxyCOR[c(24)]) #samsungdispos 
colnames(galaxyCOR[c(32)]) # htcdisneg
colnames(galaxyCOR[c(56)]) # googleperneg
colnames(galaxyCOR[c(54)]) # "googleperpos"
colnames(galaxyCOR[c(34)]) #samsungdisunc
colnames(galaxyCOR[c(19)]) # "samsungcamunc"
colnames(galaxyCOR[c(42)]) # "htcperpos"
colnames(galaxyCOR[c(21)]) # "nokiacamunc"
colnames(galaxyCOR[c(31)]) # "nokiadisneg"
colnames(galaxyCOR[c(26)]) # "nokiadispos"
colnames(galaxyCOR[c(51)]) # "nokiaperunc"
colnames(galaxyCOR[c(11)]) # "nokiacampos"
colnames(galaxyCOR[c(36)]) # "nokiadisunc"
colnames(galaxyCOR[c(46)]) # "nokiaperneg"
colnames(galaxyCOR[c(16)]) # "nokiacamneg"
colnames(galaxyCOR[c(28)]) # "iphonedisneg"
colnames(galaxyCOR[c(23)]) # "iphonedispos"
colnames(galaxyCOR[c(40)]) # "sonyperpos"
colnames(galaxyCOR[c(57)]) # "iosperunc"
colnames(galaxyCOR[c(55)]) #  "iosperneg"
colnames(galaxyCOR[c(30)]) #  sonydisneg
colnames(galaxyCOR[c(6)])  #  "ios"
colnames(galaxyCOR[c(5)])  #  "htcphone"

#################
# Feature removal
#################

# remove based on Feature Engineering (FE)
# create 34v ds
galaxyCOR34v<- galaxyCOR
galaxyCOR34v$samsungdisneg <- NULL
galaxyCOR34v$samsungperneg<- NULL
galaxyCOR34v$samsungdispos <- NULL
galaxyCOR34v$htcdisneg<- NULL
galaxyCOR34v$googleperneg<- NULL
galaxyCOR34v$googleperpos <- NULL
galaxyCOR34v$samsungdisunc <- NULL
galaxyCOR34v$samsungcamunc <- NULL
galaxyCOR34v$htcperpos <- NULL
galaxyCOR34v$nokiacamunc <- NULL
galaxyCOR34v$nokiadisneg <- NULL
galaxyCOR34v$nokiadispos <- NULL
galaxyCOR34v$nokiaperunc <- NULL
galaxyCOR34v$nokiacampos <- NULL
galaxyCOR34v$nokiadisunc <- NULL
galaxyCOR34v$nokiaperneg <- NULL
galaxyCOR34v$nokiacamneg <- NULL
galaxyCOR34v$iphonedisneg <- NULL
galaxyCOR34v$iphonedispos <- NULL
galaxyCOR34v$sonyperpos <- NULL
galaxyCOR34v$iosperunc <- NULL
galaxyCOR34v$sonydisneg <- NULL
galaxyCOR34v$iosperneg <- NULL
galaxyCOR34v$ios <- NULL
galaxyCOR34v$htcphone <- NULL
str(galaxyCOR34v)    

##################
# Train/test sets
##################
# set random seed
set.seed(123)

# create the training partition 70% of total obs
ginTrainingCOR <- createDataPartition(galaxyCOR34v$galaxysentiment, p=0.7, list=FALSE)

# create training/testing dataset
gtrainSetCOR <- galaxyCOR34v[ginTrainingCOR,]   
gtestSetCOR <- galaxyCOR34v[-ginTrainingCOR,]   

# verify number of obs 
nrow(gtrainSetCOR)
nrow(gtestSetCOR)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_FitCORg <- train(galaxysentiment~., data=gtrainSetCOR, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitCORg

#Random Forest 

#9040 samples
#33 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:

#  mtry  Accuracy   Kappa    
#2    0.7007074  0.3417145
#9    0.7514819  0.4974913
#17    0.7473444  0.4923087
#25    0.7435835  0.4873884
#33    0.7405085  0.4829996

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 9.

#--- Save top performing model ---#

saveRDS(rf_FitCORg, "rf_FitCORg.rds")  
# load and name model
rf_FitCORg <- readRDS("rf_FitCORg.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred2g <- predict(rf_FitCORg, gtestSetCOR)
rf_Pred2g
# summarize predictions
summary(rf_Pred2g)
# performance measurement
postResample(rf_Pred2g, gtestSetCOR$galaxysentiment)
#Accuracy     Kappa 
#0.7537275 0.5155909 

# calculate confusion Matrix
cmrfgcor <-confusionMatrix(rf_Pred2g,gtestSetCOR$galaxysentiment)
#######################################################
# Random Forest Model with Feature Engineering method 2
#######################################################
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 

gnzvMetrics <- nearZeroVar(galaxy_smallmatrix_N, saveMetrics = TRUE)
gnzvMetrics


# nearZeroVar() with saveMetrics = FALSE returns an vector 
gnzv <- nearZeroVar(galaxy_smallmatrix_N, saveMetrics = FALSE) 
gnzv

# create a new data set and remove near zero variance features
galaxyNZV <- galaxy_smallmatrix_N[,-gnzv]
str(galaxyNZV)

##################
# Train/test sets
##################
# set random seed
set.seed(123)
# create the training partition 70 % of total obs
ginTrainingNZV <- createDataPartition(galaxyNZV$galaxysentiment, p=0.7, list=FALSE)
# create training/testing dataset
gtrainSetNZV <- galaxyNZV[ginTrainingNZV,]   
gtestSetNZV <- galaxyNZV[-ginTrainingNZV,]   

# verify number of obs 
nrow(gtrainSetNZV)
nrow(gtestSetNZV)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats =5)

## ------- random forest ------- ##

# train/fit
set.seed(123)
grf_FitNZV <- train(galaxysentiment~., data=gtrainSetNZV, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
grf_FitNZV

#Random Forest 

#9040 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7535168  0.4971730
#4    0.7527644  0.5004511
#6    0.7489370  0.4952609
#8    0.7464813  0.4924984
#11    0.7424328  0.4867368

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 4.

#--- Save top performing model ---#

saveRDS(grf_FitNZV, "grf_FitNZV.rds")  
# load and name model
grf_FitNZV<- readRDS("grf_FitNZV.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred3g <- predict(grf_FitNZV, gtestSetNZV)
rf_Pred3g 
# summarize predictions
summary(rf_Pred3g)
# performance measurement
postResample(rf_Pred3g, gtestSetNZV$galaxysentiment)

# calculate confusion Matrix
cmrfgnzv <-confusionMatrix(rf_Pred3g,gtestSetNZV$galaxysentiment)

#######################################################
# Random Forest Model with Feature Engineering method 3
#######################################################

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxy_smallmatrix_N[sample(1:nrow(galaxy_smallmatrix_N), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResultsg <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResultsg

#--- Save top performing model ---#

saveRDS(rfeResultsg, "rfeResultsg.rds")  
# load and name model
rfeResultsg<- readRDS("rfeResultsg.rds")

# Plot results
plot(rfeResultsg, type=c("g", "o"))

# create new data set with rfe recommended features
galaxyRFE <- galaxy_smallmatrix_N[,predictors(rfeResultsg)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxy_smallmatrix_N$galaxysentiment
str(galaxyRFE)
head(galaxy_smallmatrix_N)

# review outcome
str(galaxyRFE)
summary(galaxyRFE)

##################
# Train/test sets
##################

# set random seed
set.seed(123)

# create the training partition 70 % of total obs
ginTrainingRFE <- createDataPartition(galaxyRFE$galaxysentiment, p=0.7, list=FALSE)

# create training/testing dataset
gtrainSetRFE <- galaxyRFE[ginTrainingRFE,]   
gtestSetRFE <- galaxyRFE[-ginTrainingRFE,]   

# verify number of obs 
nrow(gtrainSetRFE)
nrow(gtestSetRFE)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats =5)

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_FitRFEg <- train(galaxysentiment~., data=gtrainSetRFE, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitRFEg

#Random Forest 

#9040 samples
#25 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7233, 7232, 7232, 7232, 7232, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7171230  0.3933696
##7    0.7666145  0.5336587
#13    0.7634289  0.5308522
#19    0.7591148  0.5245968
#25    0.7552433  0.5188742

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 7.


#--- Save top performing model ---#

saveRDS(rf_FitRFEg, "rf_FitRFEg.rds")  
# load and name model
rf_FitRFEg <- readRDS("rf_FitRFEg.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred5g <- predict(rf_FitRFEg, gtestSetRFE)
rf_Pred5g
# summarize predictions
summary(rf_Pred5g)
# performance measurement
postResample(rf_Pred5g, gtestSetRFE$galaxysentiment)
# calculate confusion Matrix
cmrfeg <-confusionMatrix(rf_Pred5g,gtestSetRFE$galaxysentiment)

#####################################################################
# Random Forest Model with RFE and Engineering the Dependant variable
#####################################################################
# create a new dataset that will be used for recoding sentiment
galaxyRC <- galaxyRFE
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxyRC$galaxysentiment <- recode(galaxyRFE$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(galaxyRC)
str(galaxyRC)
# make iphonesentiment a factor
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)

##################
# Train/test sets
##################

# set random seed
set.seed(123)
ginTrainingRC <- createDataPartition(galaxyRC$galaxysentiment, p=0.7, list=FALSE)
# create training/testing dataset
gtrainSetRC <- galaxyRC[ginTrainingRC,]   
gtestSetRC <- galaxyRC[-ginTrainingRC,]   

# verify number of obs 
nrow(gtrainSetRC)
nrow(gtestSetRC)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_FitRCg <- train(galaxysentiment~., data=gtrainSetRC, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitRCg

#Random Forest 

#9039 samples
#25 predictor
#4 classes: '1', '2', '3', '4' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7231, 7232, 7230, 7231, 7232, 7232, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7958399  0.4116076
#7    0.8460444  0.6014976
#13    0.8432121  0.5965607
#19    0.8392958  0.5897275
#25    0.8369725  0.5856729

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 7.

#--- Save top performing model ---#

saveRDS(rf_FitRCg, "rf_FitRCg.rds")  
# load and name model
rf_FitRCg <- readRDS("rf_FitRCg.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred6g <- predict(rf_FitRCg, gtestSetRC)
rf_Pred6g
# summarize predictions
summary(rf_Pred6g)
# performance measurement
postResample(rf_Pred6g, gtestSetRC$galaxysentiment)
# calculate confusion Matrix
cmrcg <-confusionMatrix(rf_Pred6g, gtestSetRC$galaxysentiment, mode="everything")


#####################################################################
# Random Forest Model with PCA and Engineering the Dependent variable
#####################################################################

########### Train/test sets##############

# set random seed
set.seed(123)
# create the training partition 70 % of total obs
inTrainingggpca <- createDataPartition(galaxy_smallmatrix$galaxysentiment, p=0.7, list=FALSE)
# create training/testing dataset
gtraining <- galaxy_smallmatrix[inTrainingggpca,]   
gtesting <- galaxy_smallmatrix[-inTrainingggpca,]   
# verify number of obs 
nrow(gtraining)
nrow(gtesting)
########### recode the Dependent variable##############
gtraining$galaxysentiment <- recode(gtraining$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

gtraining$galaxysentiment <- as.factor(gtraining$galaxysentiment)

gtesting$galaxysentiment <- recode(gtesting$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

gtesting$galaxysentiment <- as.factor(gtesting$galaxysentiment)

str(gtraining)
### normalize and PCA ####
# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParamsgpca <- preProcess(gtraining[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsgpca)

# use predict to apply pca parameters, create training, exclude dependant
gtrain.pca <- predict(preprocessParamsgpca, gtraining[,-59])

# add the dependent to training
gtrain.pca$galaxysentiment <- gtraining$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
gtest.pca <- predict(preprocessParamsgpca, gtesting[,-59])

# add the dependent to training
gtest.pca$galaxysentiment <- gtesting$galaxysentiment

# inspect results
str(gtrain.pca)
str(gtest.pca)

################
# Train control
################

# set 5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_Fitpcag<- train(galaxysentiment~., data=gtrain.pca, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_Fitpcag

#Random Forest 

#9040 samples
#24 predictor
#4 classes: '1', '2', '3', '4' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7233, 7232, 7233, 7230, 7232, 7233, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.8352668  0.5757603
#7    0.8355545  0.5755580
#13    0.8352226  0.5745440
#18    0.8355764  0.5753842
#24    0.8352667  0.5745375

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 2.

##--- Save top performing model ---#

saveRDS(rf_Fitpcag, "rf_Fitpcag.rds")  
# load and name model
rf_Fitpcag <- readRDS("rf_Fitpcag.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred7g <- predict(rf_Fitpcag, gtest.pca)
rf_Pred7g
# summarize predictions
summary(rf_Pred7g)
# performance measurement
postResample(rf_Pred7g, test.pca$galaxysentiment)
#Accuracy     Kappa 
#0.8117409 0.7935469 
# calculate confusion Matrix
options(max.print=10000)
cmrf <-confusionMatrix(rf_Pred7g,gtest.pca$galaxysentiment, mode="everything")

########################################
# Feature Engineering with large matrix
########################################

# create new data set with rfe recommended features
galaxylargematrixRFE <- galaxy_largematrix_N[,predictors(rfeResultsg)]

# add the dependent variable to iphoneRFE
galaxylargematrixRFE$galaxysentiment <- galaxy_largematrix_N$galaxysentiment
str(galaxylargematrixRFE)

####################
# Predict new dataSet
####################
# predict with rf
rf_Pred8g <- predict(rf_FitRCg, galaxylargematrixRFE)
rf_Pred8g
# summarize predictions
summary(rf_Pred8g)

###############
# Save datasets
###############
galaxy_largematrixoutput <- galaxy_largematrix
galaxy_largematrixoutput$galaxysentiment<- rf_Pred8g
write.csv(galaxy_largematrixoutput, file="C:/Users/admin/galaxy_largematrixoutput.csv", row.names = TRUE)