
# Title: Sentiment analysis for Iphone in R

# Last update: 2018.10.20

# File:  sentiment analysis for Iphone.R

###############
# Project Notes
###############

# Summarize project:  We will develop several models and pick the best model using Small matrix dataset which are manually labeled with a sentiment rating for Iphone.
# We will then use the best model to predict sentiment ratings of the large matrix dataset for Iphone.

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

## Load training and Prediction set
iphone_smallmatrix<- read.csv("C:/Users/admin/iphone_smallmatrix_labeled_8d.csv", stringsAsFactors = FALSE, header=T)
iphone_largematrix<- read.csv("C:/Users/admin/iphoneLargeMatrix.csv", stringsAsFactors = FALSE, header=T)

################
# Evaluate data
################

#--- Training Set ---#
summary(iphone_smallmatrix) 
str(iphone_smallmatrix) 
head(iphone_smallmatrix)
tail(iphone_smallmatrix)

# check for missing values 
is.na(iphone_smallmatrix)
any(is.na(iphone_smallmatrix))

# plot
hist(iphone_smallmatrix$iphonesentiment)
qqnorm(iphone_smallmatrix$iphonesentiment)  

#--- Prediction Set ---# 
summary(iphone_largematrix) 
str(iphone_largematrix) 
head(iphone_largematrix)
tail(iphone_largematrix)

# check for missing values 
is.na(iphone_largematrix)
any(is.na(iphone_largematrix))

#############
# Preprocess
#############

#--- Training and test set ---#

# change variable types
iphone_smallmatrix$iphonesentiment <- as.factor(iphone_smallmatrix$iphonesentiment)
# normalize
preprocessParams <- preProcess(iphone_smallmatrix[,1:58], method = c("center", "scale"))
print(preprocessParams) 
iphone_smallmatrix_N <- predict(preprocessParams, iphone_smallmatrix)
str(iphone_smallmatrix_N)

#--- Prediction set ---#

# drop ID variable
iphone_largematrix$id<- NULL

# change variable types
iphone_largematrix$iphonesentiment <- as.factor(iphone_largematrix$iphonesentiment)

# normalize
iphone_largematrix_N <- predict(preprocessParams, iphone_largematrix)
str(iphone_largematrix_N)

##################
# Train/test sets
##################

# set random seed
set.seed(123)

# create the training partition that is 70% of total obs
inTraining <- createDataPartition(iphone_smallmatrix_N$iphonesentiment, p=0.7, list=FALSE)

# create training/testing dataset
trainSetN <- iphone_smallmatrix_N[inTraining,]   
testSetN <- iphone_smallmatrix_N[-inTraining,]   

# verify number of obs 
nrow(trainSetN)
nrow(testSetN)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

##############
# Train model
##############
## ------- Decision Tree C5.0 ------- ##

# set random seed
set.seed(123)

# train/fit
C5.0_Fit <- train(iphonesentiment~., data=trainSetN, method="C5.0", trControl=fitControl,metric = "Kappa", tuneLength=2)
C5.0_Fit

#C5.0 
#9083 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7726965  0.5583714
#rules  FALSE   10      0.7595847  0.5370933
#rules  FALSE   20      0.7595847  0.5370933
#rules  FALSE   30      0.7595847  0.5370933
#rules  FALSE   40      0.7595847  0.5370933
#rules  FALSE   50      0.7595847  0.5370933
#rules  FALSE   60      0.7595847  0.5370933
#rules  FALSE   70      0.7595847  0.5370933
#rules  FALSE   80      0.7595847  0.5370933
#rules  FALSE   90      0.7595847  0.5370933
#rules   TRUE    1      0.7724653  0.5578496
#rules   TRUE   10      0.7599037  0.5376753
#rules   TRUE   20      0.7599037  0.5376753
#rules   TRUE   30      0.7599037  0.5376753
#rules   TRUE   40      0.7599037  0.5376753
#rules   TRUE   50      0.7599037  0.5376753
#rules   TRUE   60      0.7599037  0.5376753
#rules   TRUE   70      0.7599037  0.5376753
#rules   TRUE   80      0.7599037  0.5376753
#rules   TRUE   90      0.7599037  0.5376753
#tree   FALSE    1      0.7723770  0.5585957
#tree   FALSE   10      0.7600357  0.5402329
#tree   FALSE   20      0.7600357  0.5402329
#tree   FALSE   30      0.7600357  0.5402329
#tree   FALSE   40      0.7600357  0.5402329
#tree   FALSE   50      0.7600357  0.5402329
#tree   FALSE   60      0.7600357  0.5402329
#tree   FALSE   70      0.7600357  0.5402329
#tree   FALSE   80      0.7600357  0.5402329
#tree   FALSE   90      0.7600357  0.5402329
#tree    TRUE    1      0.7718708  0.5576906
#tree    TRUE   10      0.7593423  0.5388084
#tree    TRUE   20      0.7593423  0.5388084
#tree    TRUE   30      0.7593423  0.5388084
#tree    TRUE   40      0.7593423  0.5388084
#tree    TRUE   50      0.7593423  0.5388084
#tree    TRUE   60      0.7593423  0.5388084
#tree    TRUE   70      0.7593423  0.5388084
#tree    TRUE   80      0.7593423  0.5388084
#tree    TRUE   90      0.7593423  0.5388084

#Kappa was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = tree and winnow = FALSE.

#--- Save top performing model ---#

saveRDS(C5.0_Fit, "C5.0_Fit.rds")  
# load and name model
C5.0_Fit <- readRDS("C5.0_Fit.rds")

#---Predict testSet with  C5.0---#

# predict with C5.0
C5.0_Pred1 <- predict(C5.0_Fit, testSetN)
C5.0_Pred1 

#summarize predictions
summary(C5.0_Pred1)

# performance measurement
postResample(C5.0_Pred1, testSetN$iphonesentiment)
#Accuracy     Kappa 
#0.7665810 0.5460934
# plot
plot(C5.0_Pred1,testSetN$iphonesentiment)
#calculate confusion Matrix
options(max.print=10000)
cmC5.0 <- confusionMatrix(C5.0_Pred1, testSetN$iphonesentiment, mode="everything")

## ------- random forest ------- ##
# set random seed
set.seed(123)
# train/fit
rf_Fit <- train(iphonesentiment~., data=trainSetN, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=10) 
rf_Fit

#Random Forest 

#9083 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7020362  0.3757150
#8    0.7754054  0.5621092
#14    0.7764945  0.5664788
#20    0.7760432  0.5664790
#26    0.7749199  0.5652814
#33    0.7726188  0.5622435
#39    0.7697012  0.5580462
#45    0.7686773  0.5568668
#51    0.7665964  0.5536154
#58    0.7644274  0.5504335

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 20.

#--- Save top performing model ---#

saveRDS(rf_Fit, "rf_Fit.rds")  
# load and name model
rf_Fit <- readRDS("rf_Fit.rds")

#---Predict testSet with  rf---#

# predict with rf
rf_Pred1 <- predict(rf_Fit, testSetN)
rf_Pred1 
#summarize predictions
summary(testSetN$iphonesentiment)
summary(rf_Pred1) 

# performance measurement
postResample(rf_Pred1, testSetN$iphonesentiment)
#Accuracy     Kappa 
#0.7750643 0.5632066 

#calculate confusion Matrix
cmrf <-confusionMatrix(rf_Pred1,testSetN$iphonesentiment,mode="everything")
cmrf

## ------- SVM ------- ##

## ------- SVM Linear ------- ##
# set random seed
set.seed(123)
# train/fit
svm_Fit <- train(iphonesentiment~., data=trainSetN, method="svmLinear", trControl=fitControl, metric = "Kappa",tuneLength=5) 
svm_Fit

#Support Vector Machines with Linear Kernel 

#9083 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results:
  
#  Accuracy   Kappa    
#0.7111868  0.4191533

#Tuning parameter 'C' was held constant at a value of 1

#--- Save top performing model ---#

saveRDS(svm_Fit, "svmFit.rds")  
# load and name model
svm_Fit<- readRDS("svmFit.rds")

## ------- SVM Radial ------- ##

# set random seed
set.seed(123)
# train/fit
svm_Fitr <- train(iphonesentiment~., data=trainSetN, method="svmRadial", trControl=fitControl, metric = "Kappa",tuneLength=5) 
svm_Fitr

#Support Vector Machines with Radial Basis Function Kernel 

#9083 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#  C     Accuracy   Kappa    
#0.25  0.7201273  0.4254658
#0.50  0.7306853  0.4493697
#1.00  0.7381494  0.4701697
#2.00  0.7356832  0.4684343
#4.00  0.7334923  0.4660332

#Tuning parameter 'sigma' was held constant at a value of 3.343925
#Kappa was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 3.343925 and C = 1.

#--- Save top performing model ---#

saveRDS(svm_Fitr, "svmFitr.rds")  
# load and name model
svm_Fitr<- readRDS("svmFitr.rds")

#---Predict testSet with svm---#

# predict with svm
svmPred1 <- predict(svm_Fit, testSetN)
# print predictions
svmPred1
#summarize predictions
summary(svmPred1)

# performance measurement
postResample(svmPred1, testSetN$iphonesentiment)

#calculate confusion Matrix
cmsvm <-confusionMatrix(svmPred1, testSetN$iphonesentiment)


## ------- KKNN------- ##

# set random seed
set.seed(123)

# KKNN train/fit
kknn_Fit <- train(iphonesentiment~., data=trainSetN, method="kknn", trControl=fitControl, metric = "Kappa",tuneLength=5) 
kknn_Fit

#k-Nearest Neighbors 

#9083 samples
#58 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#  kmax  Accuracy   Kappa    
#5    0.3093578  0.1523941
#7    0.3203896  0.1576053
#9    0.3296931  0.1618064
#11    0.3397889  0.1670506
#13    0.3460640  0.1706907

#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal
#Kappa was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 13, distance = 2 and kernel = optimal.


#--- Save top performing model ---#

saveRDS(kknn_Fit, "kknnFit.rds")  
# load and name model
kknn_Fit<- readRDS("kknnFit.rds")

#---Predict testSet with kknn---#

# predict with kknn
kknnPred1 <- predict(kknn_Fit, testSetN)
# print predictions
kknnPred1
#summarize predictions
summary(kknnPred1)
#performance measurement
postResample(kknnPred1, testSetN$iphonesentiment)
#Accuracy     Kappa 
#0.3462725 0.1744809 

#calculate confusion Matrix
cmkknn <-confusionMatrix(kknnPred1, testSetN$iphonesentiment)
cmkknn

## ------- GBM------- ##

#recode variable of training set
trainSetNG <-trainSetN
trainSetNG$iphonesentiment <- recode(trainSetNG$iphonesentiment, '0' = 'seg0', '1' = 'seg1', '2' = 'seg2', '3' = 'seg3', '4' = 'seg4', '5' = 'seg5')
str(trainSetNG)

#recode variable of test set
testSetNG <-testSetN
testSetNG$iphonesentiment <- recode(testSetNG$iphonesentiment, '0' = 'seg0', '1' = 'seg1', '2' = 'seg2', '3' = 'seg3', '4' = 'seg4', '5' = 'seg5')
str(testSetNG)

# set random seed
set.seed(123)
# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

# GBM train/fit
model3<- train(iphonesentiment~., data=trainSetNG,method='gbm',trControl = fitControl,verbose=F,tuneLength=3)

model3

# Stochastic Gradient Boosting 

# 9083 samples
# 58 predictor
# 6 classes: 'seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5' 

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
# Resampling results across tuning parameters:
  
#  interaction.depth  n.trees  Accuracy   Kappa    
# 1                   50      0.7306735  0.4603539
# 1                  100      0.7414631  0.4858461
# 1                  150      0.7483551  0.5017653
# 2                   50      0.7495113  0.5034272
# 2                  100      0.7627223  0.5358695
# 2                  150      0.7715955  0.5583070
# 3                   50      0.7659480  0.5434859
# 3                  100      0.7728179  0.5608181
# 3                  150      0.7731814  0.5619847

# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Tuning parameter 'n.minobsinnode' was held
# constant at a value of 10
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode
# = 10.

#--- Save top performing model ---#

saveRDS(model3, "gbmFit3.rds")  
# load and name model
gbm_Fit3<- readRDS("gbmFit3.rds")

#---Predict testSet---#

# predict with gbm
gbmPred1 <- predict(gbm_Fit3, testSetNG)
# print predictions
gbmPred1
# summarize predictions
summary(gbmPred1)

# performance measurement
postResample(gbmPred1, testSetNG$iphonesentiment)
#Accuracy     Kappa 
#0.7712082 0.557140

#calculate confusion Matrix
cmgbm <-confusionMatrix(gbmPred1, testSetNG$iphonesentiment,mode="everything")

##--- Compare metrics ---##

ModelFitResults <- resamples(list(C5.0=C5.0_Fit,rf=rf_Fit,SVM=svm_Fit,kknn=kknn_Fit,SVMr=svm_Fitr,gbm=gbm_Fit3))
# output summary metrics for tuned models 
summary(ModelFitResults)

# Call:
#   summary.resamples(object = ModelFitResults)

# Models: C5.0, rf, SVM, kknn, SVMr, gbm 
# Number of resamples: 100 

# Accuracy 
#  Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.7535754 0.7656121 0.7719008 0.7723770 0.7779919 0.8019802    0
# rf   0.7568757 0.7689133 0.7754540 0.7760432 0.7822137 0.8063806    0
# SVM  0.6864686 0.7037445 0.7114534 0.7111868 0.7181392 0.7403740    0
# kknn 0.3142227 0.3361808 0.3472755 0.3460640 0.3558249 0.3817382    0
# SVMr 0.7120879 0.7305573 0.7370737 0.7381494 0.7442821 0.7665198    0
# gbm  0.7524752 0.7673458 0.7733773 0.7731814 0.7795535 0.8030803    0

# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.5173203 0.5446247 0.5578421 0.5585957 0.5715318 0.6240833    0
# rf   0.5245796 0.5505577 0.5653320 0.5664790 0.5787580 0.6346380    0
# SVM  0.3633234 0.4036823 0.4194541 0.4191533 0.4353608 0.4852688    0
# kknn 0.1354306 0.1596399 0.1720956 0.1706907 0.1788448 0.2086101    0
# SVMr 0.4055091 0.4533903 0.4674321 0.4701697 0.4847049 0.5375705    0
# gbm  0.5171666 0.5480796 0.5622067 0.5619847 0.5754679 0.6280857    0

#######################################################
# Random Forest Model with Feature Engineering method 1
#######################################################

options(max.print=1000000)
iphoneCOR <- iphone_smallmatrix_N
corrAll <- cor(iphoneCOR[,1:59])
corrAll 

# plot correlation matrix
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
corr58<- cor(iphoneCOR[,1:58])

# create object with indexes of highly corr features
corrIVhigh <- findCorrelation(corr58, cutoff=0.8)
# print indexes of highly correlated attributes
corrIVhigh

# get var name of high corr IV
colnames(iphoneCOR[c(29)]) #samsungdisneg
colnames(iphoneCOR[c(44)]) #samsungperneg
colnames(iphoneCOR[c(24)]) #samsungdispos 
colnames(iphoneCOR[c(32)]) # htcdisneg
colnames(iphoneCOR[c(56)]) # googleperneg
colnames(iphoneCOR[c(54)]) # "googleperpos"
colnames(iphoneCOR[c(34)]) #samsungdisunc
colnames(iphoneCOR[c(19)]) # "samsungcamunc"
colnames(iphoneCOR[c(42)]) # "htcperpos"
colnames(iphoneCOR[c(21)]) # "nokiacamunc"
colnames(iphoneCOR[c(31)]) # "nokiadisneg"
colnames(iphoneCOR[c(26)]) # "nokiadispos"
colnames(iphoneCOR[c(51)]) # "nokiaperunc"
colnames(iphoneCOR[c(11)]) # "nokiacampos"
colnames(iphoneCOR[c(36)]) # "nokiadisunc"
colnames(iphoneCOR[c(46)]) # "nokiaperneg"
colnames(iphoneCOR[c(16)]) # "nokiacamneg"
colnames(iphoneCOR[c(28)]) # "iphonedisneg"
colnames(iphoneCOR[c(23)]) # "iphonedispos"
colnames(iphoneCOR[c(25)]) # "sonydispos"
colnames(iphoneCOR[c(57)]) # "iosperunc"
colnames(iphoneCOR[c(55)]) #  "iosperneg"
colnames(iphoneCOR[c(6)])  #  "ios"
colnames(iphoneCOR[c(5)])  #  "htcphone"

#---Feature removal---#

# remove based on Feature Engineering (FE)
# create 34v ds
iphoneCOR34v<- iphoneCOR
iphoneCOR34v$samsungdisneg <- NULL
iphoneCOR34v$samsungperneg<- NULL
iphoneCOR34v$samsungdispos <- NULL
iphoneCOR34v$htcdisneg<- NULL
iphoneCOR34v$googleperneg<- NULL
iphoneCOR34v$googleperpos <- NULL
iphoneCOR34v$samsungdisunc <- NULL
iphoneCOR34v$samsungcamunc <- NULL
iphoneCOR34v$htcperpos <- NULL
iphoneCOR34v$nokiacamunc <- NULL
iphoneCOR34v$nokiadisneg <- NULL
iphoneCOR34v$nokiadispos <- NULL
iphoneCOR34v$nokiaperunc <- NULL
iphoneCOR34v$nokiacampos <- NULL
iphoneCOR34v$nokiadisunc <- NULL
iphoneCOR34v$nokiaperneg <- NULL
iphoneCOR34v$nokiacamneg <- NULL
iphoneCOR34v$iphonedisneg <- NULL
iphoneCOR34v$iphonedispos <- NULL
iphoneCOR34v$sonydispos <- NULL
iphoneCOR34v$iosperunc <- NULL
iphoneCOR34v$iosperneg <- NULL
iphoneCOR34v$ios <- NULL
iphoneCOR34v$htcphone <- NULL
str(iphoneCOR34v)   

##################
# Train/test sets
##################

# set random seed
set.seed(123)
# create the training partition 70% of total obs 
inTraining <- createDataPartition(iphoneCOR34v$iphonesentiment, p=0.7, list=FALSE)

# create training/testing dataset
trainSetCOR <- iphoneCOR34v[inTraining,]   
testSetCOR <- iphoneCOR34v[-inTraining,]   

# verify number of obs 
nrow(trainSetCOR)
nrow(testSetCOR)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

##############
# Train model
##############

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_FitCOR <- train(iphonesentiment~., data=trainSetCOR, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitCOR

#Random Forest 

#9083 samples
#34 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 7267, 7269, 7266, 7265, 7265, 7266, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.6932742  0.3509564
#10    0.7548172  0.5209051
#18    0.7517568  0.5176004
#26    0.7480136  0.5126151
#34    0.7441165  0.5072790

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 10.

#--- Save top performing model ---#

saveRDS(rf_FitCOR, "rf_FitCOR.rds")  
# load and name model
rf_FitCOR <- readRDS("rf_FitCOR.rds")

#################
# Predict testSet
#################
# predict with rf
rf_Pred2 <- predict(rf_FitCOR, testSetCOR)
rf_Pred2 
# summarize predictions
summary(rf_PPred2)
# performance measurement
postResample(rf_Pred2, testSetCOR$iphonesentiment)
#Accuracy     Kappa 
#0.7537275 0.5155909 

# calculate confusion Matrix
cmrfc <-confusionMatrix(rf_Pred2,testSetCOR$iphonesentiment,mode="everything")
#######################################################
# Random Forest Model with Feature Engineering method 2
#######################################################
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 

nzvMetrics <- nearZeroVar(iphone_smallmatrix_N, saveMetrics = TRUE)
nzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iphone_smallmatrix_N, saveMetrics = FALSE) 
nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphone_smallmatrix_N[,-nzv]
str(iphoneNZV)

##################
# Train/test sets
##################
# set random seed
set.seed(123)
# create the training partition 70 % of total obs
inTrainingNZV <- createDataPartition(iphoneNZV$iphonesentiment, p=0.7, list=FALSE)
# create training/testing dataset
trainSetNZV <- iphoneNZV[inTrainingNZV,]   
testSetNZV <- iphoneNZV[-inTrainingNZV,]   

# verify number of obs 
nrow(trainSetNZV)
nrow(testSetNZV)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

##############
# Train model
##############

## ------- random forest ------- ##

# train/fit
set.seed(123)
rf_FitNZV <- train(iphonesentiment~., data=trainSetNZV, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitNZV

#Random Forest 

#9083 samples
#11 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7606732  0.5282633
#4    0.7612230  0.5324151
#6    0.7584486  0.5293810
#8    0.7550575  0.5248470
#11    0.7506312  0.5189007

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 4.

#--- Save top performing model ---#

saveRDS(rf_FitNZV, "rf_FitNZV.rds")  
# load and name model
rf_FitNZV<- readRDS("rf_FitNZV.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred3 <- predict(rf_FitNZV, testSetNZV)
rf_Pred3 
# summarize predictions
summary(rfPred3)
# performance measurement
postResample(rf_Pred3, testSetNZV$iphonesentiment)
# calculate confusion Matrix
cmrfnzv <-confusionMatrix(rf_Pred3,testSetNZV$iphonesentiment,mode="everything")

#######################################################
# Random Forest Model with Feature Engineering method 3
#######################################################

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphone_smallmatrix_N[sample(1:nrow(iphone_smallmatrix_N), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

#--- Save top performing model ---#

saveRDS(rfeResults, "rfeResults.rds")  
# load and name model
rfeResults <- readRDS("rfeResults.rds")

# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- iphone_smallmatrix_N[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone_smallmatrix_N$iphonesentiment
head(iphoneRFE)
head(iphone_smallmatrix_N)

# review outcome
str(iphoneRFE)
summary(iphoneRFE)
summary(trainSetN)

##################
# Train/test sets
##################

# set random seed
set.seed(123)

# create the training partition 70 % of total obs
inTraining <- createDataPartition(iphoneRFE$iphonesentiment, p=0.7, list=FALSE)

# create training/testing dataset
trainSetRFE <- iphoneRFE[inTraining,]   
testSetRFE <- iphoneRFE[-inTraining,]   

# verify number of obs 
nrow(trainSetRFE)
nrow(testSetRFE)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

## ------- random forest ------- ##

# set random seed
set.seed(123)

# train/fit

rf_FitRFE <- train(iphonesentiment~., data=trainSetRFE, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitRFE

#Random Forest 

#9083 samples
#26 predictor
#6 classes: '0', '1', '2', '3', '4', '5' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8175, 8175, 8174, 8175, 8175, 8175, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7148843  0.4095846
#8    0.7767257  0.5675275
#14    0.7734776  0.5637519
#20    0.7688092  0.5569119
#26    0.7649891  0.5513335
#
#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 8.

#--- Save top performing model ---#

saveRDS(rf_FitRFE, "rf_FitRFE.rds")  
# load and name model
rf_FitRFE <- readRDS("rf_FitRFE.rds")

#################
# Predict testSet
#################
# predict with rf
rf_Pred5 <- predict(rf_FitRFE, testSetRFE)
rf_Pred5

# summarize predictions
summary(rfPred5)
# performance measurement
postResample(rf_Pred5, testSetRFE$iphonesentiment)

# calculate confusion Matrix
cmrfref <-confusionMatrix(rf_Pred5,testSetRFE$iphonesentiment, mode="everything")

#####################################################################
# Random Forest Model with RFE and Engineering the Dependant variable
#####################################################################

# create a new dataset that will be used for recoding sentiment
iphoneRC <- iphoneRFE
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRFE$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphoneRC)
str(iphoneRC)
# make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)
str(iphoneRC)

##################
# Train/test sets
##################

# set random seed
set.seed(123)

# create the training partition 70 % of total obs
inTraining <- createDataPartition(iphoneRC$iphonesentiment, p=0.7, list=FALSE)
# create training/testing dataset
trainSetRC <- iphoneRC[inTraining,]   
testSetRC <- iphoneRC[-inTraining,]   

# verify number of obs 
nrow(trainSetRC)
nrow(testSetRC)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

## ------- random forest ------- ##

# train/fit
set.seed(123)
rf_FitRC <- train(iphonesentiment~., data=trainSetRC, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_FitRC

#Random Forest 

#9083 samples
#27 predictor
#4 classes: '1', '2', '3', '4' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8174, 8176, 8175, 8176, 8175, 8174, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.7886284  0.4178696
#8    0.8485857  0.6222858
#14    0.8483874  0.6232784
#20    0.8462299  0.6194118
#27    0.8437747  0.6150385

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 14.


#--- Save top performing model ---#

saveRDS(rf_FitRC, "rf_FitRC.rds")  
# load and name model
rf_FitRC <- readRDS("rf_FitRC.rds")

#################
# Predict testSet
#################
# predict with rf
rf_Pred6 <- predict(rf_FitRC, testSetRC)
rf_Pred6
# summarize predictions
summary(rf_Pred6)
summary(testSetRC$iphonesentiment)
# performance measurement
postResample(rf_Pred6, testSetRC$iphonesentiment)

#Accuracy     Kappa 
#0.8519280 0.6335108 

# calculate confusion Matrix
cmrc <-confusionMatrix(rf_Pred6, testSetRC$iphonesentiment, mode = "everything")

#####################################################################
# Random Forest Model with PCA and Engineering the Dependent variable
#####################################################################

########### Train/test sets##############

# set random seed
set.seed(123)

# create the training partition 70 % of total obs
inTrainingpca <- createDataPartition(iphone_smallmatrix$iphonesentiment, p=0.7, list=FALSE)

# create training/testing dataset
training <- iphone_smallmatrix[inTrainingpca,]   
testing <- iphone_smallmatrix[-inTrainingpca,]   
# verify number of obs 
nrow(training)
nrow(testing)


########### recode the Dependent variable##############
training$iphonesentiment <- recode(training$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

training$iphonesentiment <- as.factor(training$iphonesentiment)

testing$iphonesentiment <- recode(testing$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

testing$iphonesentiment <- as.factor(testing$iphonesentiment)

str(training)

### normalize and PCA ####
# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, training[,-59])

# add the dependent to training
train.pca$iphonesentiment <- training$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testing[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testing$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

## ------- random forest ------- ##

# train/fit

set.seed(123)
rf_Fitpca<- train(iphonesentiment~., data=train.pca, method="rf", trControl=fitControl, metric = "Kappa",tuneLength=5) 
rf_Fitpca

#Random Forest 

#9083 samples
#25 predictor
#4 classes: '1', '2', '3', '4' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 8174, 8176, 8175, 8176, 8175, 8174, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.8414197  0.6079849
#7    0.8428179  0.6108781
#13    0.8425317  0.6100802
#19    0.8420690  0.6089492
#25    0.8413315  0.6069969

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 7.

#--- Save top performing model ---#

saveRDS(rf_Fitpca, "rf_Fitpca.rds")  
# load and name model
rf_Fitpca <- readRDS("rf_Fitpca.rds")
#################
# Predict testSet
#################
# predict with rf
rf_Pred7 <- predict(rf_Fitpca, test.pca)
rf_Pred7
# summarize predictions
summary(rfPred7)
# performance measurement
postResample(rf_Pred7, test.pca$iphonesentiment)
#Accuracy     Kappa 
#0.8383033 0.5980302

# calculate confusion Matrix
cmpca <-confusionMatrix(rf_Pred7,test.pca$iphonesentiment,mode = "everything")
cmpca

########################################
# Feature Engineering with large matrix
########################################

# create new data set with rfe recommended features
iphonelargematrixRFE <- iphone_largematrix_N[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphonelargematrixRFE$iphonesentiment <- iphone_largematrix_N$iphonesentiment
str(iphonelargematrixRFE)

####################
# Predict new dataSet
####################
# predict with rf
rf_Pred8 <- predict(rf_FitRC, iphonelargematrixRFE)
rf_Pred8

# summarize predictions
summary(rf_Pred8)

#1     2     3     4 
#9152   928  1232 11736 

###############
# Save datasets
###############
iphone_largematrixoutput <- iphone_largematrix
iphone_largematrixoutput$iphonesentiment<- rf_Pred8
write.csv(iphone_largematrixoutput, file="C:/Users/admin/iphone_largematrixoutput.csv", row.names = TRUE)
