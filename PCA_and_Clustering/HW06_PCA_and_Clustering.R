# read data into memory
data_set <- read.csv("data_set.csv", header = TRUE)

# get X and y values
X <- as.matrix(subset(scale(data_set), select = -c(Type) )) 
y <- data_set$Type

# get number of samples and number of features
N <- length(y)
D <- ncol(X)

# calculate the covariance matrix
Sigma_X <- cov(X)

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(Sigma_X, symmetric = TRUE)

# plot scree graph
plot(1:D, decomposition$values, 
     type = "l", las = 1, lwd = 2,
     xlab = "Eigenvalue index", ylab = "Eigenvalue")

# plot proportion of variance explained
pove <- cumsum(decomposition$values) / sum(decomposition$values)
plot(1:D, pove, 
     type = "l", las = 1, lwd = 2,
     xlab = "R", ylab = "Proportion of variance explained")
abline(h = 0.90, lwd = 2, lty = 2, col = "blue")
abline(v = which(pove > 0.90)[1], lwd = 2, lty = 2, col = "blue")

# calculate two-dimensional projections
Z <- (X - matrix(colMeans(X), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:3]

#printing first 4 rows of Z matrix
print(Z[1:4,])

# plot two-dimensional projections
#PC1 & PC2
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a")
plot(Z[,1], Z[,2], type = "p", col = point_colors[y], xlab = "PC1", ylab = "PC2", cex = 1, axes = TRUE, pch = 19)
legend("bottomright",c("BRCA","COAD","KIRC","LUAD","PRAD"),fill=point_colors)

#PC1 & PC3
plot(Z[,1], Z[,3], type = "p", col = point_colors[y], xlab = "PC1", ylab = "PC3", cex = 1, axes = TRUE, pch = 19)
legend("bottomright",c("BRCA","COAD","KIRC","LUAD","PRAD"),fill=point_colors)

#PC2 & PC3
plot(Z[,2], Z[,3], type = "p", col = point_colors[y], xlab = "PC2", ylab = "PC3", cex = 1, axes = TRUE, pch = 19)
legend("bottomleft",c("BRCA","COAD","KIRC","LUAD","PRAD"),fill=point_colors)



# create new variables for K-means algorithm
set.seed(421)
centroids <<- NULL
old_centroids <<- NULL
assignments <<- NULL
Z_k <- Z[,c(1,3)]
K <- 5
max_iteration <- 10

# Start K-means algotithm
for(i in 1:max_iteration){
  old_centroids <<- centroids
  
  if (is.null(centroids) == TRUE) {
    centroids <- Z_k[sample(1:N, K),]
  } else {
    for (k in 1:K) {
      centroids[k,] <- colMeans(Z_k[assignments == k,])
    }
  }
  
  D_k <- as.matrix(dist(rbind(centroids, Z_k), method = "euclidean"))
  D_k <- D_k[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(Z_k))]
  assignments <<- sapply(1:ncol(D_k), function(c) {which.min(D_k[,c])})

  if(old_centroids == centroids && i > 1)  
    break
}

print(centroids)

# plot two-dimensional projections and centroids
point_colors <- c("#ff7f00", "#e31a1c", "#33a02c", "#6a3d9a", "#1f78b4")
plot(Z_k[,1], Z_k[,2], col = point_colors[assignments], xlab = "PC1", ylab = "PC3", cex = 1, axes = TRUE, pch = 19)
points(centroids[,1], centroids[,2], col = "black", pch = 19, cex = 2)
legend("bottomright",c("Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5"),fill=point_colors)
