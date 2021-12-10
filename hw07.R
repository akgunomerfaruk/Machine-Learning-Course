library(caret)
library(FactoMineR)
library(factoextra)
library(mice)
library(car)
library(xgboost)
library(AUC)

set.seed(421)

#read data into memory

  X_train1 <- read.csv("hw07_target1_training_data.csv",  header = TRUE)
  Y_train1 <- read.csv("hw07_target1_training_label.csv", header = TRUE)
  X_test1  <- read.csv("hw07_target1_test_data.csv",      header = TRUE) 
  
  X_train2 <- read.csv("hw07_target2_training_data.csv",  header = TRUE)
  Y_train2 <- read.csv("hw07_target2_training_label.csv", header = TRUE)
  X_test2  <- read.csv("hw07_target2_test_data.csv",      header = TRUE) 
  
  X_train3 <- read.csv("hw07_target3_training_data.csv",  header = TRUE)
  Y_train3 <- read.csv("hw07_target3_training_label.csv", header = TRUE)
  X_test3  <- read.csv("hw07_target3_test_data.csv",      header = TRUE)

# DATA PREPERATION (ENCODING, IMPUTATION & FEATURE SELECTION)
  
#one-hot-encoding categoricals & imputation using a tree based method - Training 1
  dmy1 <- dummyVars(" ~ .", data = X_train1)
  X_train1_d <- data.frame(predict(dmy1, newdata = X_train1))
  X_train1_c <- mice(X_train1_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_train1_clean <- data.matrix(complete(X_train1_c, action = 1L), rownames.force = TRUE)
  write.csv(X_train1_clean, file="X_train1_clean.csv")
  any(is.na(X_train1_clean))

#there are many highly corrolated features in Training 1 
  corMatrix1  <- cor(X_train1_clean[])
  corrolated1 <- findCorrelation(corMatrix1, cutoff = 0.80)

#one-hot-encoding categoricals & imputation using a tree based method - Training 2
  dmy2 <- dummyVars(" ~ .", data = X_train2)
  X_train2_d <- data.frame(predict(dmy2, newdata = X_train2))
  X_train2_c <- mice(X_train2_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_train2_clean <- data.matrix(complete(X_train2_c, action = 1L))
  write.csv(X_train2_clean, file="X_train2_clean.csv")
  any(is.na(X_train2_clean))

#there are many highly corrolated features in Training 2
  corMatrix2  <- cor(X_train2_clean[])
  corrolated2 <- findCorrelation(corMatrix2, cutoff = 0.80)
  
#one-hot-encoding categoricals & imputation using a tree based method Training 3  
  dmy3 <- dummyVars(" ~ .", data = X_train3)
  X_train3_d <- data.frame(predict(dmy3, newdata = X_train3))
  X_train3_c <- mice(X_train3_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_train3_clean <- data.matrix(complete(X_train3_c, action = 1L))
  write.csv(X_train3_clean, file="X_train3_clean.csv")
  any(is.na(X_train3_clean))

#there are many highly corrolated features in Training 3
  corMatrix3  <- cor(X_train3_clean[])
  corrolated3 <- findCorrelation(corMatrix3, cutoff = 0.80)
  
#one-hot-encoding categoricals & imputation using a tree based method - Test Sets
  dmy11 <- dummyVars(" ~ .", data = X_test1)
  X_test1_d <- data.frame(predict(dmy11, newdata = X_test1))
  X_test1_c <- mice(X_test1_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_test1_clean <- data.matrix(complete(X_test1_c, action = 1L))
  any(is.na(X_test1_clean))
  write.csv(X_test1_clean, file="X_test1_clean.csv")
  
  dmy22 <- dummyVars(" ~ .", data = X_test2)
  X_test2_d <- data.frame(predict(dmy22, newdata = X_test2))
  X_test2_c <- mice(X_test2_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_test2_clean <- data.matrix(complete(X_test2_c, action = 1L))
  any(is.na(X_test2_clean))
  write.csv(X_test2_clean, file="X_test2_clean.csv")
  
  dmy33 <- dummyVars(" ~ .", data = X_test3)
  X_test3_d <- data.frame(predict(dmy33, newdata = X_test3))
  X_test3_c <- mice(X_test3_d, meth="cart", m=1, maxit=1, remove_collinear=FALSE)
  X_test3_clean <- data.matrix(complete(X_test3_c, action = 1L))
  any(is.na(X_test3_clean))
  write.csv(X_test3_clean, file="X_test3_clean.csv")

#DIMENSIONALITY REDUCTION - NOT USED 
  pca1 <- PCA(X_train1_clean, scale=TRUE)
  fviz_eig(pca1)
  get_eigenvalue(pca1)
  
  pca2 <- PCA(X_train2_clean, scale=TRUE)
  fviz_eig(pca2)
  get_eigenvalue(pca2)
  
  pca3 <- PCA(X_train3_clean, scale=TRUE)
  fviz_eig(pca3) 
  get_eigenvalue(pca3)

#TRAINING
  
#grid for parameter tuning
  xg_grid <- expand.grid(
    nrounds = 30, 
    max_depth = c(3, 6, 10), 
    eta = 0.3, 
    gamma = c(0, 1, 2), 
    colsample_bytree = c(0.4, 0.7, 1.0), 
    min_child_weight = c(0.5, 1, 1.5),
    subsample=c(0.5,1)
  )
  
  xg_grid23 <- expand.grid(
    nrounds = 150, 
    max_depth = 3, 
    eta = 0.3, 
    gamma = 0, 
    colsample_bytree = 0.7, 
    min_child_weight = 0.5,
    subsample=1
  )
#control parameters for train - method, number of folds and parallel Set #1
  xgtrcontrol <- trainControl(
    verboseIter = TRUE,
    classProbs=TRUE,
    method = "cv",
    number = 5,
    allowParallel = TRUE,
  )
  
  xgtrcontrol23 <- trainControl(
    verboseIter = TRUE,
    classProbs=TRUE,
    method = "cv",
    number = 20,
    allowParallel = TRUE,
  )
  
#train the models
xg_model_tuned1 <- train(
    X_train1_clean[, -1],
    Y_train1[,2],
    trControl = xgtrcontrol,
    objective = "binary:logistic",
    eval_metric = "auc",
    tuneGrid = xg_grid,
    method = "xgbTree"
  )

xg_model_tuned11 <- train(
  X_train1_clean[, -1],
  Y_train1[,2],
  trControl = xgtrcontrol23,
  objective = "binary:logistic",
  eval_metric = "auc",
  tuneGrid = xg_grid23,
  method = "xgbTree"
)

xg_model_tuned2 <- train(
  X_train2_clean[, -1],
  Y_train2[,2],
  trControl = xgtrcontrol23,
  objective = "binary:logistic",
  eval_metric = "auc",
  method = "xgbTree",
  tuneGrid = xg_grid23
  )

xg_model_tuned3 <- train(
  X_train3_clean[, -1],
  Y_train3[,2],
  trControl = xgtrcontrol23,
  objective = "binary:logistic",
  eval_metric = "auc",
  method = "xgbTree",
  tuneGrid = xg_grid23
  
)

#PREDICTING

#Training predictions
training_scores1 <- predict(xg_model_tuned11, X_train1_clean[, -1])
training_scores2 <- predict(xg_model_tuned2, X_train2_clean[, -1])
training_scores3 <- predict(xg_model_tuned3, X_train3_clean[, -1])

# AUC score for training data
print(auc(roc(predictions = training_scores1, labels = as.factor(Y_train1[, "TARGET"]))))
print(auc(roc(predictions = training_scores2, labels = as.factor(Y_train2[, "TARGET"]))))
print(auc(roc(predictions = training_scores3, labels = as.factor(Y_train3[, "TARGET"]))))

#Test predictions
test_scores1 <- predict(xg_model_tuned11, X_test1_clean[, -1])
test_scores2 <- predict(xg_model_tuned2, X_test2_clean[, -1])
test_scores3 <- predict(xg_model_tuned3, X_test3_clean[, -1])

#Test csv files
write.table(test_scores1, file = "hw07_target1_test_predictions.csv", row.names = FALSE)
write.table(test_scores2, file = "hw07_target2_test_predictions.csv", row.names = FALSE)
write.table(test_scores3, file = "hw07_target3_test_predictions.csv", row.names = FALSE)
