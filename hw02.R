# read data into memory
dataset_labels <- read.csv("hw02_labels.csv", header = FALSE)
dataset_images <- read.csv("hw02_images.csv", header = FALSE)
initial_w0 <- read.csv("initial_w0.csv", header = FALSE)
initial_W  <- read.csv("initial_W.csv",  header = FALSE)

#define safelog
safelog <- function(x) {log(x + 1e-100)}

# define sigmoid
sigmoid <- function(X, W, w0) {
  A <- X %*% W
  sapply (X=1:K,FUN = function(j) 1/ (1 + exp(-(A[, j]+ w0[j]))))
  }

#split into two 
X_train <- data.matrix(dataset_images [1:500, ])
X_test  <- data.matrix(dataset_images [501:1000, ])
y_train <- c(dataset_labels [1:500, ])
y_test  <- c(dataset_labels [501:1000, ])

# keep K and N
K<- max(y_train)
N<-length(y_train)

#one-of-K-encoding
Y_train <-matrix(0, N, K)
Y_train[cbind(1:N, y_train)] <- 1

# define gradients
gradient_W <- function(X, Y_truth, Y_hat) {
  sapply(X = 1:ncol(Y_truth), function(j) colSums(matrix((Y_truth[,j] - Y_hat[,j]) * Y_hat[,j] * (1-Y_hat[,j]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X))
}

gradient_w0 <- function(Y_truth, Y_hat) {
  colSums((Y_truth - Y_hat) * Y_hat * (1-Y_hat))
}

#initialize W & w0
W <- data.matrix(initial_W)
w0 <- data.matrix(initial_w0)

# set learning parameters
eta <- 0.0001
epsilon <- 1e-3
max_iteration <- 500


# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  
  #training data posteriors
  Y_hat_train <- sigmoid (X_train , W , w0)
  
  sqerror <- sum(0.5*(Y_train-Y_hat_train)^2)
  
  objective_values <- c(objective_values, sqerror)
  
  W_old <- W
  w0_old <- w0
  
  #update W and w0
  W <- W + eta * gradient_W(X_train, Y_train, Y_hat_train)
  w0 <- w0 + eta * gradient_w0(Y_train, Y_hat_train)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
  
  if (iteration > max_iteration-1) {
    break
  }

}

#keep predicted classes by checking posteriors
y_predicted_train <- sapply(X=1:N, FUN = function(n) {
  which.max(Y_hat_train[n,])
})

#training data confusion matrix
confusion_train <- table(y_predicted_train,y_train)

#test data posteriors
Y_hat_test <- sigmoid(X_test, W, w0)

#test data classes
y_predicted_test <- sapply(X=1:N, FUN = function(n) {
  which.max(Y_hat_test[n,])
})

#test data confusion matrix
confusion_test <- table(y_predicted_test, y_test)

#error troughout the iterations
plot(1:499, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

print(confusion_train)
print(confusion_test)