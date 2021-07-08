# read data into memory
data_set <- read.csv("data_set.csv")

# get X and y values
X <- data_set[, 1:209]
y <- data_set$Type

# get number of classes and number of features
K <- max(y)
D <- ncol(X)



# get numbers of train and test samples
N_train <- length(y_train)
N_test <- length(y_test)

# create necessary data structures
node_indices <- list()
is_terminal <- c()
need_split <- c()

node_features <- c()
node_splits <- c()
node_frequencies <- list()

# put all training instances into the root node
node_indices <- list(1:N_train)
is_terminal <- c(FALSE)
need_split <- c(TRUE)


# learning algorithm
while (1) {
  # find nodes that need splitting
  split_nodes <- which(need_split)
  # check whether we reach all terminal nodes
  if (length(split_nodes) == 0) {
    break
  }
  # find best split positions for all nodes
  for (split_node in split_nodes) {
    data_indices <- node_indices[[split_node]]
    need_split[split_node] <- FALSE
    node_frequencies[[split_node]] <- sapply(1:K, function(c) {sum(y_train[data_indices] == c)})
    # check whether node is pure
    if (length(unique(y_train[data_indices])) == 1) {
      is_terminal[split_node] <- TRUE
    } else {
      is_terminal[split_node] <- FALSE
      
      best_scores <- rep(0, D)
      best_splits <- rep(0, D)
      for (d in 1:D) {
        unique_values <- sort(unique(X_train[data_indices, d]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(X_train[data_indices, d] < split_positions[s])]
          right_indices <- data_indices[which(X_train[data_indices, d] >= split_positions[s])]
          split_scores[s] <- -length(left_indices) / length(data_indices) * sum(sapply(1:K, function(c) {mean(y_train[left_indices] == c) * log2(mean(y_train[left_indices] == c))}), na.rm = TRUE) +
            -length(right_indices) / length(data_indices) * sum(sapply(1:K, function(c) {mean(y_train[right_indices] == c) * log2(mean(y_train[right_indices] == c))}), na.rm = TRUE)
        }
        best_scores[d] <- min(split_scores)
        best_splits[d] <- split_positions[which.min(split_scores)]
      }
      # decide where to split on which feature
      split_d <- which.min(best_scores)
      node_features[split_node] <- split_d
      node_splits[split_node] <- best_splits[split_d]
      
      # create left node using the selected split
      left_indices <- data_indices[which(X_train[data_indices, split_d] < best_splits[split_d])]
      node_indices[[2 * split_node]] <- left_indices
      is_terminal[2 * split_node] <- FALSE
      need_split[2 * split_node] <- TRUE
      
      # create left node using the selected split
      right_indices <- data_indices[which(X_train[data_indices, split_d] >= best_splits[split_d])]
      node_indices[[2 * split_node + 1]] <- right_indices
      is_terminal[2 * split_node + 1] <- FALSE
      need_split[2 * split_node + 1] <- TRUE
    }
  }
}


# extract rules
terminal_nodes <- which(is_terminal)
for (terminal_node in terminal_nodes) {
  index <- terminal_node
  rules <- c()
  while (index > 1) {
    parent <- floor(index / 2)
    if (index %% 2 == 0) {
      # if node is left child of its parent
      rules <- c(sprintf("x%d < %g", node_features[parent], node_splits[parent]), rules)
    } else {
      # if node is right child of its parent
      rules <- c(sprintf("x%d >= %g", node_features[parent], node_splits[parent]), rules)
    }
    index <- parent
  }
  print(sprintf("{%s} => [%s]", paste0(rules, collapse = " AND "), paste0(node_frequencies[[terminal_node]], collapse = "-")))
}



# traverse tree for train data points
y_predicted <- rep(0, N_train)
for (i in 1:N_train) {
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      y_predicted[i] <- which.max(node_frequencies[[index]])
      break
    } else {
      if (X_train[i, node_features[index]] < node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}
confusion_matrix <- table(y_predicted, y_train)
print(confusion_matrix)



# traverse tree for test data points
y_predicted <- rep(0, N_test)
for (i in 1:N_test) {
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      y_predicted[i] <- which.max(node_frequencies[[index]])
      break
    } else {
      if (X_test[i, node_features[index]] < node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}
confusion_matrix <- table(y_predicted, y_test)
print(confusion_matrix)
