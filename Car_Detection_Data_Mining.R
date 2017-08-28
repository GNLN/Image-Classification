install.packages("pixmap")
library("pixmap")

install.packages("xgboost")
library(xgboost)

install.packages("naivebayes")
library(naivebayes)
library(e1071)

library("readxl")
library(dummies)
library(dplyr)
library(datasets)
library(arules)
library(arulesViz)
library(stringr)
library(caret)
library(ROCR)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@data cleaning@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 

filenames <- list.files("C:/Users/sathy/Downloads/Spring '17/Data Minning/Project/TrainImages", pattern = "*.pgm", full.names = TRUE)
result={}
for (i in filenames){
  x <- read.pnm(file = i)
  y <- getChannels(x)
  y <- as.vector(y)
  result <- rbind(result,y)
}

#extarcting negative or positive for training
newfilenames <- gsub("C:/Users/sathy/Downloads/Spring '17/Data Minning/Project/TrainImages/*", "", filenames)
modfilenames <- gsub("*.pgm", "", newfilenames)
tt <- substr(modfilenames, start = 1, stop = 3)
new <- ifelse(tt == "pos", 1, 0) 


df <- as.data.frame.matrix(result)
df$result <- new

hist(df$V8)

rownames(df)<- 1:nrow(df)

################testing data###################
rownames(df_test)<-1:nrow(df_test)
###############################################

#breaking data into training and validation
set.seed(123457)

#dividing to trainig and testing dataset
train<-sample(nrow(df),0.8*nrow(df))
df_train <-df[train,]
df_validation <-df[-train,]

write.csv(df_train.pca, file = "df_train_pca.csv")
write.csv(df_validation.pca, file = "df_validation_pca.csv")



#PCA-screeplot does not work for this control object, I will give that graph when making ppt
control<-preProcess(df_train[,-4001], method=c("BoxCox", "center","pca"))
df_train.pca<-predict(control, df_train[,-4001])
print(control)
df_train.pca$result <- df_train$result

df_validation.pca<- predict(control, df_validation[,-4001])
df_validation.pca$result <- df_validation$result
screeplot(ctrl)

##################testing dataset#######################
dim(df_test.pca)<- predict(control, df_test[,-4001])
df_test.pca$result <- df_test$result
#######################################################



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$logistic regression$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

system.time(logfit <- glm(result~., data=df_train.pca, family= "binomial"))
logfit
p <- predict(logfit, newdata= df_train.pca, type = "response")
p

#predicting for the validation data set
p_validation <-predict(logfit, newdata = df_validation.pca, type = "response")
p_validation


#*****************************************************
system.time(logfit_normal <- glm(result~., data=df_train, family= "binomial"))
logfit_normal
p_normal <- predict(logfit_normal, newdata= df_train, type = "response")
p_normal

#predicting for the validation data set
p_validation_normal <-predict(logfit_normal, newdata = df_validation, type = "response")
p_validation_normal

df_validation$LOGPREDICTION <- p_validation_normal
df_validation$LOGPREFCLASS <- ifelse(df_validation$LOGPREDICTION>=0.5, 1, 0)
normal_lr <-confusionMatrix(df_validation$result, df_validation$LOGPREFCLASS)
normal_lr

accuracy_n_lr <- (normal_lr[1,1]+normal_lr[2,2])/(normal_lr[1,1]+normal_lr[2,1]+normal_lr[1,2]+normal_lr[2,2])
accuracy_n_lr


#*****************************************************

######################testing predcition####################
p_test <-predict(logfit, newdata = df_test.pca, type = "response")
p_test
testresults <- ifelse(p_test>=0.5, 1, 0)
###########################################################


df_validation.pca$LOGPREDICTION <- p_validation
df_validation.pca$LOGPREFCLASS <- ifelse(df_validation.pca$LOGPREDICTION>=0.5, 1, 0)
logconfmat_valid <-confusionMatrix(df_validation.pca$result, df_validation.pca$LOGPREFCLASS)
logconfmat_valid

accuracy <- (logconfmat_valid[1,1]+logconfmat_valid[2,2])/(logconfmat_valid[1,1]+logconfmat_valid[2,1]+logconfmat_valid[1,2]+logconfmat_valid[2,2])
accuracy


class(df_train.pca$result)


#sensitivity training
Sensitivity <- logconfmat_valid[2,2]/(logconfmat_valid[2,1]+logconfmat_valid[2,2])
Sensitivity
#specificity training
specificity <- logconfmat_valid[1,1]/(logconfmat_valid[1,2]+logconfmat_valid[1,1])
specificity

#PPV and NVP train 
PPV <- logconfmat_valid[2,2]/(logconfmat_valid[1,2]+logconfmat_valid[2,2])
NPV <- logconfmat_valid[1,1]/(logconfmat_valid[2,1]+logconfmat_valid[1,1])
PPV
NPV


#rocr for logistic regression

pred_test_lr<- prediction(df_validation.pca$LOGPREDICTION, df_validation.pca$result)
perf_test_lr<- performance(pred_test, "tpr", "fpr")
x <-plot(perf_test_lr, col = "RED")
legend("bottomright", legend=c("Logistic Regression"), col=c("red"), lty=1:2, title='Legend', bty='n', cex=0.65)

#maximum accuracy
max_acc_t_lr <- max(perf_test_lr@y.values[[1]])
cutoff_list_acc_lr <- unlist(perf_test_lr@x.values[[1]])
opt_cutoff_acc_lr <- cutoff_list_acc_lr[which.max(perf_test_lr@y.values[[1]])]

pred_lr <- ifelse(df_validation.pca$LOGPREDICTION > opt_cutoff_acc_lr, 1,0)
confusionMatrix(df_validation.pca$result,pred_lr)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NaiveBayes~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Naive Bayes

model <- naiveBayes(as.factor(result)~.,data=df_train.pca)
model
prediction_naive <- predict(model, newdata = df_validation.pca)
prediction_naive1 <- predict(model, newdata = df_validation.pca,'raw')

a <- data.frame(prediction_naive1)

#confusion matrix
confusion_nb <- confusionMatrix(prediction_naive,df_validation.pca$result)

#ROC Curve

pred_test<- prediction(a$X1, df_validation.pca$result)
perf_test<- performance(pred_test, "tpr", "fpr")
x <-plot(perf_test, color="green")
max_acc_t <- max(perf_test@y.values[[1]])
cutoff_list_acc <- unlist(perf_test@x.values[[1]])
opt_cutoff_acc <- cutoff_list_acc[which.max(perf_test@y.values[[1]])]


# Using the optimal cut-off value
pred_nb <- ifelse(a$X1> opt_cutoff_acc, 1,0)
confusionMatrix(df_validation.pca$result,pred_nb)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Boosting%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#XGBOOST
xgdfprepca <- df_train
result = df_train$result
xgdfprepca$result <- NULL
xgdfprepca = as.matrix(xgdfprepca)
system.time(bst <- xgboost(data = xgdfprepca, label = result, max.depth = 2, eta = 1, nround = 5, objective = "binary:logistic"))

#Predictions with model built
xgdfprepcaval <- df_validation
actual <- xgdfprepcaval$result
xgdfprepcaval$result <- NULL
xgpred <- predict(bst, newdata = as.matrix(xgdfprepcaval))
xgpred <- ifelse(xgpred>0.5,1,0)
(boostingconfmat_valid.prepca <-confusionMatrix(actual, xgpred))

#Performing XGBoost after PCA and selecting 404 components.
xgdf <- df_train.pca
result = xgdf$result
xgdf$result <- NULL
xgdf = as.matrix(xgdf)
bst <- xgboost(data = xgdf, label = result, max.depth = 2, eta = 1, nround = 5, objective = "binary:logistic")
system.time(bst <- xgboost(data = xgdf, label = result, max.depth = 2, eta = 1, nround = 5, objective = "binary:logistic"))

#Predictions with model built
xgdfval <- df_validation.pca
actual <- xgdfval$result
xgdfval$result <- NULL
xgpred <- predict(bst, newdata = as.matrix(xgdfval))
xgpred <- ifelse(xgpred>0.5,1,0)
(boostingconfmat_valid.pca <-confusionMatrix(actual, xgpred))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^plots^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
data = df[900, 1:4000]
df[200, 4001]
oo = data.matrix(data)
barplot(oo, main="pixel shade chart", xlab="pixels", ylab = "shade") 
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^























