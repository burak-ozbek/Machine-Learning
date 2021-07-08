#read data
data("iris")
X_iris_data <- as.matrix(iris[,1:4])
y_iris_vector <- as.numeric(iris[,5])


#collect first 25 rows for each species 
X <- rbind(X_iris_data[1:25,],X_iris_data[51:75,],X_iris_data[101:125,])

y_train <- c(y_iris_vector[1:25],y_iris_vector[51:75],y_iris_vector[101:125])

y_truth <- y_train

# get number of classes and number of samples
K <- max(y_truth)
N <- length(y_truth)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

safelog <- function(X) {
  return (log(X + 1e-100))
}


sigmoid <- function(X,W, w0){
  return(sapply(X=1:ncol(W), function(c) (matrix(1/ (1+ exp(-cbind(X,1)%*% rbind(W,w0) [,c])), nrow=nrow(X)/K, ncol=ncol(W), byrow=FALSE))))
}

# define the gradient functions
gradient_W <- function(X, Y_truth, Y_predicted) {
  return (sapply(X = 1:ncol(Y_truth), function(c) -colSums(matrix((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1-Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
  }

gradient_w0 <- function(Y_truth, Y_predicted) {
  return ( colSums(sapply(X = 1:ncol(Y_truth), function(c) -colSums(matrix((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1-Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE)))) )
}


# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize W and w0
set.seed(421)
W <- matrix(runif(ncol(X) * K, min = -0.001, max = 0.001), ncol(X), K)
w0 <- runif(K, min = -0.001, max = 0.001)

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  Y_predicted <- sigmoid(X, W, w0)
  objective_values <- c(objective_values, 0.5 * sum((Y_predicted - Y_truth)^2))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(w0)


# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


# calculate confusion matrix
y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)
