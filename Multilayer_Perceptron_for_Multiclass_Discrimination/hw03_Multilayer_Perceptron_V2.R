safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data into memory
X_data <- read.csv("hw03_digits.csv")
y_data <- read.csv("hw03_labels.csv")


class_size <- 100
class_count <- 10
ratio <- 0.8

X_train <- NULL
y_truth <- NULL
Xtest <- NULL
y_test <- NULL

iteration <- 1
# seperate data as it wanted
while(1){
  
  #train data set
  st <- (iteration-1)*class_size +1
  en <- (iteration-1)*class_size + class_size * ratio
  
  X_train <- rbind(X_train, X_data[st:en,])
  y_truth <- c(y_truth,y_data[st:en,])
  
  
  #test data set
  st <- (iteration-1)*class_size + class_size * ratio +1
  en <- iteration * class_size 
  
  Xtest <- rbind(Xtest, X_data[st:en,])
  y_test <- c(y_test,y_data[st:en,])
  
  iteration <- iteration +1
  if (iteration > class_count){
    break
  }
}

X <- X_train

# get number of classes  number of samples and number of features
K <- max(y_truth)
N <- length(y_truth)
D <- ncol(X)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1


# define the sigmoid function
sigmoid <- function(X, W){
  return(1/ (1+ exp(-as.matrix(cbind(1, X))%*%as.matrix(W))) )
}


# define the softmax function
softmax <- function(Z, V) {
  scores <- as.matrix(cbind(1, Z)) %*% as.matrix(V)
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}


# set learning parameters
eta <- 0.0005 
epsilon <- 1e-3
H <- 20 
max_iteration <- 500 


# define the gradient functions
gradient_delta_v <- function(X, Y_truth, Y_predicted) {
  return (sapply(X = 1:ncol(Y_truth), function(c) -colSums(matrix((Y_truth[,c]-Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_delta_W <- function(X,Z,Y_truth, Y_predicted) {
  return ( - t(cbind(1,X)) %*%  (rowSums(sapply(X = 1:ncol(Y_truth), function(c) (Y_truth[,c] - Y_predicted[,c])%*% t(V[2:(H + 1),c]))) * (Z[, 1:H] * (1 - Z[, 1:H])))  )
}


# randomly initalize W and v
set.seed(421)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H) 
V <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H + 1, K)

#initialize first predicted values
Z <- sigmoid(X, W)
Y_predicted <- softmax(Z, V)
objective_values <- - sum(Y_truth * safelog(Y_predicted))

#set iteration
iteration <- 1
while (1) {
  iteration <- iteration + 1
  
  delta_v <- eta * gradient_delta_v(cbind(1,Z),Y_truth,Y_predicted) 
  
  delta_W <- eta * gradient_delta_W(X,Z,Y_truth,Y_predicted)
  
  V <- V - delta_v
  W <- W - delta_W
  
  # calculate hidden nodes
  Z <- sigmoid(X, W)
  
  # calculate output node
  Y_predicted <- softmax(Z, V)
  
  objective_values <- c(objective_values, -sum(Y_truth * safelog(Y_predicted)))
  
  if ( abs(objective_values[iteration - 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
}
#print(W)
#print(V)

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


# calculate confusion matrix for the train data points
y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)



#calculate predictions for test data points with using the trained multilayer perceptron
Z <- sigmoid(Xtest, W)
Y_predicted <- softmax(Z, V)

# calculate confusion matrix for the test data points
y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_test)
print(confusion_matrix)
