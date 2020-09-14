rm(list = ls())

# Load the libraries
library(xgboost)
library(dplyr)
library(Matrix)
library(data.table)
library(ggplot2)

# Load the dataset
load_csv_data <- function(csv_file, sample_ratio = 1, drop_cols = NULL,
                          sel_cols = NULL) {
  
  dt <- fread(csv_file, header = TRUE, sep = ",", stringsAsFactors = TRUE,
              na.strings = "", drop = drop_cols, select = sel_cols,
              showProgress = TRUE)
  
  if (sample_ratio < 1) {
    rows <- nrow(dt)
    sample_size <- as.integer(sample_ratio * rows)
    dt <- dt[sample(.N, sample_size)]
  }
  
  return(dt)
}

# One hot encoding function
# No character allowed
one_hot_sparse <- function(data_set) {
  
  # Numerical data to sparse matrix
  out_put_data <- data_set[,sapply(data_set, is.numeric), with = FALSE]
  out_put_data <- as(as.matrix(out_put_data), "dgCMatrix")
  
  # Logical to sparse and join the matrix
  out_put_data <- cbind2(out_put_data,
                         as(as.matrix(data_set[,sapply(data_set,
                                                       is.logical),
                                               with = FALSE]), "dgCMatrix"))
  
  # Identify factor columns
  fact_variables <- names(which(sapply(data_set, is.factor)))
  
  # One hot encoding for each column
  i <- 0
  
  for (f_var in fact_variables) {
    
    f_col_names <- levels(data_set[[f_var]])
    f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
    j_values <- as.numeric(data_set[[f_var]]) 
    
    if (sum(is.na(j_values)) > 0) {  
      j_values[is.na(j_values)] <- length(f_col_names) + 1
      f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
    }
    
    if (i == 0) {
      fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                x = rep(1, nrow(data_set)),
                                dims = c(nrow(data_set), length(f_col_names)))
      fact_data@Dimnames[[2]] <- f_col_names
    } else {
      fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                    x = rep(1, nrow(data_set)),
                                    dims = c(nrow(data_set), length(f_col_names)))
      fact_data_tmp@Dimnames[[2]] <- f_col_names
      fact_data <- cbind(fact_data, fact_data_tmp)
    }
    
    i <- i + 1
  }
  
  if (length(fact_variables) > 0) {
    out_put_data <- cbind(out_put_data, fact_data)
  }
  
  return(out_put_data)
}



# Set working directory, separated from testing csv
setwd(...)

files <- list.files(pattern = "*.csv")

# Read 100% of the datasets
df <- do.call(rbind, lapply(files, load_csv_data))

# Choose the variables to keep
# Had to delete them due to confidential data
VARS_TO_KEEP <- c(...)


# Rows with 0 (no click) and 1 (click)
df_0 <- which(df$Label == "0")
df_1 <- which(df$Label == "1")

# Take the shorter one
nsamp <- min(length(df_1), length(df_0))

# We pick all 1s and sample 0s to balance the set (5% 1 - 95% 0)
pick_1 <- sample(df_1, nsamp)
pick_0 <- sample(df_0, 19*nsamp)

new_df <- df[c(pick_1, pick_0), ]

# Remove old df
rm(df)

# Check the balance
prop.table(table(new_df$Label))

#Set working directory to test working directory
setwd(...)

# Load the test dataset
files_test <- load_csv_data("ctr_test.csv")

# Join the dataframes
df_test <- rbind(new_df, files_test, fill = TRUE)

rm(files_test)
rm(new_df)

# One hot encoding
df_one_hot <- one_hot_sparse(df_test) 

rm(df_test)

#Identify the training, validation and evaluation
tr_index <- which(!is.na(df_one_hot[, colnames(df_one_hot) == "Label"]))

# Choose 20% randomly for validation
vd_index <- sample(tr_index, 0.2 * length(tr_index))
ev_index <- which(is.na(df_one_hot[, colnames(df_one_hot) == "Label"]))

eval_data <- df_one_hot[ev_index, colnames(df_one_hot) != "Label"]

# XGBoost format
dtrain <- xgb.DMatrix(data = df_one_hot[setdiff(tr_index, vd_index),
                                        colnames(df_one_hot) != "Label"],
                      label = df_one_hot[setdiff(tr_index, vd_index),
                                         colnames(df_one_hot) == "Label"])

dvalid <- xgb.DMatrix(data = df_one_hot[vd_index, colnames(df_one_hot) != "Label"],
                      label = df_one_hot[vd_index, colnames(df_one_hot) == "Label"])

rm(df_one_hot)

# Train model
watchlist <- list(train=dtrain, test=dvalid)

# Best result model -> AUC = 0.874227
vanilla_model <- xgb.train(data=dtrain, nrounds=5000,
                           watchlist = watchlist,
                           objective = "binary:logistic",  
                           eval.metric = "auc", max_depth = 10,
                           print_every_n = 1, early_stopping_rounds=100, eta = .15, 
                           scale_pos_weight = balance)

# Feature Importance
head(xgb.importance(model=vanilla_model), 50)

# Predict and save predictions in csv
preds <- predict(vanilla_model, eval_data)
ids <- eval_data[, colnames(eval_data) == "id"]
options(scipen = 999)
write.table(data.frame(id = ids, pred = preds), file = "modelo.csv",
            sep = ",", row.names=FALSE, quote=FALSE)


# Grid search for hyperparameters

random_grid <- function(size,
                        min_nrounds, max_nrounds,
                        min_max_depth, max_max_depth,
                        min_eta, max_eta,
                        min_gamma, max_gamma,
                        min_colsample_bytree, max_colsample_bytree,
                        min_min_child_weight, max_min_child_weight,
                        min_subsample, max_subsample) {
  
  rgrid <- data.frame(nrounds = sample(c(min_nrounds:max_nrounds),
                                       size = size, replace = TRUE),
                      max_depth = sample(c(min_max_depth:max_max_depth),
                                         size = size, replace = TRUE),
                      eta = round(runif(size, min_eta, max_eta), 5),
                      gamma = round(runif(size, min_gamma, max_gamma), 5),
                      colsample_bytree = round(runif(size, min_colsample_bytree,
                                                     max_colsample_bytree), 5),
                      min_child_weight = round(runif(size, min_min_child_weight,
                                                     max_min_child_weight), 5),
                      subsample = round(runif(size, min_subsample, max_subsample), 5))
  return(rgrid)    
}

rgrid <- random_grid(size = 70,
                     min_nrounds = 50, max_nrounds = 150,
                     min_max_depth = 2, max_max_depth = 14,
                     min_eta = 0.001, max_eta = 0.15,
                     min_gamma = 0, max_gamma = 1,
                     min_colsample_bytree = 0.5, max_colsample_bytree = 1,
                     min_min_child_weight = 0, max_min_child_weight = 2,
                     min_subsample = 0.5, max_subsample = 1)

train_xgboost <- function(data_train, data_val, rgrid) {
  
  watchlist <- list(train = data_train, valid = data_val)
  
  predicted_models <- list()
  
  for (i in seq_len(nrow(rgrid))) {
    print(i)
    print(rgrid[i,])
    trained_model <- xgb.train(data = data_train,
                               params=as.list(rgrid[i, c("max_depth",
                                                         "eta",
                                                         "gamma",
                                                         "colsample_bytree",
                                                         "subsample",
                                                         "min_child_weight")]),
                               nrounds = rgrid[i, "nrounds"],
                               watchlist = watchlist,
                               objective = "binary:logistic",
                               eval.metric = "auc", eval.metric = "error",
                               print_every_n = 10)
    
    perf_tr <- tail(trained_model$evaluation_log, 1)$train_auc
    perf_vd <- tail(trained_model$evaluation_log, 1)$valid_auc
    print(c(perf_tr, perf_vd))
    
    predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                       perf_tr = perf_tr,
                                                       perf_vd = perf_vd),
                                  model = trained_model)
    rm(trained_model)
    gc()
  }
  
  return(predicted_models)
}

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)

# Results
result_table <- function(pred_models) {
  res_table <- data.frame()
  i <- 1
  for (m in pred_models) {
    res_table <- rbind(res_table, data.frame(i = i, m$results))
    i <- i + 1
  }
  res_table <- res_table[order(-res_table$perf_vd),]
  return(res_table)
}

res_table <- result_table(predicted_models) 
print(res_table)



####### LightGBM #######

rm(list = ls())

library(Matrix)
library(data.table)
library(lightgbm)


# Set working directory to training directory
files <- list.files(pattern = "*.csv")
df <- do.call(rbind, lapply(files, load_csv_data))

# Choose training and test set
smp_size <- floor(.8 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
training_set <- df[train_ind, ]
testing_set <- df[-train_ind, ]

# LightGBM format
train_df <- lgb.prepare(data = training_set)
test_df <- lgb.prepare(data = testing_set)

rm(df)

my_data_train <- as.matrix(train_df[, 2:52, with = FALSE])
my_data_test <- as.matrix(test_df[, 2:52, with = FALSE])

dtrain <- lgb.Dataset(data = my_data_train,
                      label = train_df$Label)

dtest <- lgb.Dataset.create.valid(dtrain, data = my_data_test, label = test_df$Label)

# Train with validation set
valids <- list(train = dtrain, test = dtest)

modelo <- lgb.train(data = dtrain,objective = "binary", eval = c("auc", "binary_logloss"), valids = valids,
                    nrounds = 500,learning_rate = .05, max_depth = 10, num_leaves = 255, min_data_in_leaf = 25)

# Set working directory to testing directory
files_test <- load_csv_data("ctr_test.csv")

# Prepare testing set in LightGBM format
test <- lgb.prepare(data = files_test)

my_test <- as.matrix(test[, 1:51, with = FALSE])

# Prediction with LightGBM
prediction <- predict(modelo, my_test)

summary(prediction)

# Save predictions
preds <- data.table(id=files_test$id, pred=prediction)
fwrite(preds, "light_gbm.csv")

