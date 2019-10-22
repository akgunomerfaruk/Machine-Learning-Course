# read data into memory
dataset_labels <- read.csv("hw01_labels.csv", header = FALSE)
dataset_images <- read.csv("hw01_images.csv", header = FALSE)

# combine image and label data set
dataset_comb <- cbind(dataset_images, dataset_labels)

#split the dataset into two (train and test)
bound = 200
trainset <- dataset_comb[1:bound, ]
testset  <- dataset_comb[(bound+1):nrow(dataset_comb), ]

#split the trainset into two (males and females)
females_train <- subset(trainset, trainset[, 4097]== 1)
males_train   <- subset(trainset,trainset[,4097] ==2)

#calculate sample means and keep it in a list
sample_means1 <- c(sapply(females_train[, 1:4096], mean))
sample_means2 <- c(sapply(males_train[, 1:4096], mean))
sample_means  <- cbind(sample_means1, sample_means2)

#calculate sample deviations & diagonal covariance matrices (triangels are zero since we are doing naive bayes)
sdpop <- function(x) { sqrt(mean((x-mean(x))^2)) }

sample_deviations1 <- sapply(females_train[, 1:4096], sdpop)
sample_deviations2 <- sapply(males_train[, 1:4096], sdpop)
sample_deviations  <- cbind(sample_deviations1, sample_deviations2)

sample_varience1 <- sample_deviations1*sample_deviations1
sample_varience2 <- sample_deviations2*sample_deviations2

covariance_matrix1 <- diag(sample_varience1)
covariance_matrix2 <- diag(sample_varience2)


#calculate priors and list
prior1 <- nrow(females_train)/nrow(trainset)
prior2 <- nrow(males_train)/nrow(trainset)
priors <- c(prior1, prior2)


#build score function
Wd  <- -0.5*chol2inv(chol(covariance_matrix1)) -(-0.5*chol2inv(chol(covariance_matrix2)))
wd  <- chol2inv(chol(covariance_matrix1)) %*% sample_means1 - chol2inv(chol(covariance_matrix2)) %*% sample_means2
wd0 <- -0.5 * t(sample_means1) %*% chol2inv(chol(covariance_matrix1)) %*% sample_means1 - 0.5*sum(log(sample_varience1))+ log(prior1) -(-0.5 * t(sample_means2) %*% chol2inv(chol(covariance_matrix2)) %*% sample_means2 - 0.5*sum(log(sample_varience2))+ log(prior2))

score_function <- function(X) {t(X)%*%Wd%*%X + t(wd)%*%X+ wd0}

#evaluate score values
scores_train <- apply(trainset[ ,1:4096], 1, FUN = score_function)
scores_test  <- apply(testset[ ,1:4096], 1, FUN = score_function)

#confusion_matrices
confusion_train <- table (trainset [, 4097], scores_train<0)
confusion_test  <- table (testset  [, 4097], scores_test<0)

print(sample_means[, 1])
print(sample_means[, 2])
print(sample_deviations[, 1])
print(sample_deviations[, 2])
print(confusion_train)
print(confusion_test)


