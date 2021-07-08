# read data into memory
data_set <- read.csv("forest_cover_data.csv")

# get X and y values
X <- subset(data_set, select = -c(Cover_Type) )
y <- data_set$Cover_Type

# get train and test splits
train_ratio <- 0.8
set.seed(421)

train_indices <-  c(sample(which(y == 1), floor(sum(y == 1) * train_ratio)), 
                    sample(which(y == 2), floor(sum(y == 2) * train_ratio)), 
                    sample(which(y == 3), floor(sum(y == 3) * train_ratio)))

X_train <- X[train_indices,]
X_test <- X[-train_indices,]

# get number of classes and number of features
K <- max(y)
D <- nrow(X_train)

Y_train <- NULL

for (i in 1:K){
  Y_train <- as.matrix(cbind(Y_train,(2 * (y[train_indices] == i) - 1)))
}

colnames(Y_train) <- c("1", "2", "3")

# define Euclidean distance function
pdist <- function(X1, X2) {
  if (identical(X1, X2) == TRUE) {
    D <- as.matrix(dist(X1))
  }
  else {
    D <- as.matrix(dist(rbind(X1, X2)))
    D <- D[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))]
  }
  return(D)
}

# define Gaussian kernel function
gaussian_kernel <- function(X1, X2, s) {
  D <- pdist(X1, X2)
  K <- exp(-D^2 / (2 * s^2))
}

s <- 5
K_train <- gaussian_kernel(X_train, X_train, s)

# set learning parameters
C <- 10
epsilon <- 1e-3
Y_predicted <- NULL
Alpha <- matrix(NA, nrow = D, ncol = K)
W0 <- matrix(NA, nrow = 1, ncol = K)

# add library required to solve QP problems
library(kernlab)
for (i in 1:K)
{
  # get number of samples and number of features
  N_train <- length(Y_train[,i])
  D_train <- ncol(X_train)
  
  yyK <- (Y_train[,i] %*% t(Y_train[,i])) * K_train
  
  result <- ipop(c = rep(-1, N_train), H = yyK,
                 A = Y_train[,i], b = 0, r = 0,
                 l = rep(0, N_train), u = rep(C, N_train))
  alpha <- result@primal
  alpha[alpha < C * epsilon] <- 0
  alpha[alpha > C * (1 - epsilon)] <- C
  
  # find bias parameter
  support_indices <- which(alpha != 0)
  active_indices <- which(alpha != 0 & alpha < C)
  w0 <- mean(Y_train[active_indices,i] * (1 - yyK[active_indices, support_indices] %*% alpha[support_indices]))
  
  Alpha[,i] <- alpha
  W0[,i] <- w0
  
  # calculate predictions on training samples
  f_predicted <- K_train %*% (Y_train[,i] * alpha) + w0
  
  Y_predicted <- cbind(Y_predicted,2 * (f_predicted > 0) - 1)
  print(i)
  
}

# calculate predictions on train samples
Confusion_matrix <- lapply(1:K, function(c) table(Y_predicted[,c], Y_train[,c]))
print(Confusion_matrix)




# calculate predictions on test samples
F_test_predicted <- NULL
y_test <- y[-train_indices]
K_test <- gaussian_kernel(X_test, X_train, s)
F_test_predicted <- sapply(1:K, function(c) {K_test %*% (Alpha[,c] * Y_train[,c]) + W0[,c]})
y_test_predicted <- apply(F_test_predicted, MARGIN = 1, FUN = which.max)
Confusion_test_matrix <-table(y_test_predicted, y_test)
print(Confusion_test_matrix)
