# Install packages
library(keras)
library(reticulate)
library(dplyr)
#keras::install_tensorflow()
#install_keras()


# Read data
data <- read.csv("../data/new_clean.csv")
str(data)

data = data %>% mutate(s1s2= ifelse(data$s1 * data$s2 == 0, 0, ifelse(data$s1 * data$s2 > 0, sqrt(data$s1 * data$s2), -sqrt(-(data$s1 * data$s2)))),
                       s1s3= ifelse(data$s1 * data$s3 == 0, 0, ifelse(data$s1 * data$s3 > 0, sqrt(data$s1 * data$s3), -sqrt(-(data$s1 * data$s3)))),
                       s1s4= ifelse(data$s1 * data$s4 == 0, 0, ifelse(data$s1 * data$s4 > 0, sqrt(data$s1 * data$s4), -sqrt(-(data$s1 * data$s4)))),
                       s1s5= ifelse(data$s1 * data$s5 == 0, 0, ifelse(data$s1 * data$s5 > 0, sqrt(data$s1 * data$s5), -sqrt(-(data$s1 * data$s5)))),
                       s1s6= ifelse(data$s1 * data$s6 == 0, 0, ifelse(data$s1 * data$s6 > 0, sqrt(data$s1 * data$s6), -sqrt(-(data$s1 * data$s6)))),
                       s1s7= ifelse(data$s1 * data$s7 == 0, 0, ifelse(data$s1 * data$s7 > 0, sqrt(data$s1 * data$s7), -sqrt(-(data$s1 * data$s7)))),
                       s1s8= ifelse(data$s1 * data$s8 == 0, 0, ifelse(data$s1 * data$s8 > 0, sqrt(data$s1 * data$s8), -sqrt(-(data$s1 * data$s8)))),
                       s2s3= ifelse(data$s2 * data$s3 == 0, 0, ifelse(data$s2 * data$s3 > 0, sqrt(data$s2 * data$s3), -sqrt(-(data$s2 * data$s3)))),
                       s2s4= ifelse(data$s2 * data$s4 == 0, 0, ifelse(data$s2 * data$s4 > 0, sqrt(data$s2 * data$s4), -sqrt(-(data$s2 * data$s4)))),
                       s2s5= ifelse(data$s2 * data$s5 == 0, 0, ifelse(data$s2 * data$s5 > 0, sqrt(data$s2 * data$s5), -sqrt(-(data$s2 * data$s5)))),
                       s2s6= ifelse(data$s2 * data$s6 == 0, 0, ifelse(data$s2 * data$s6 > 0, sqrt(data$s2 * data$s6), -sqrt(-(data$s2 * data$s6)))),
                       s2s7= ifelse(data$s2 * data$s7 == 0, 0, ifelse(data$s2 * data$s7 > 0, sqrt(data$s2 * data$s7), -sqrt(-(data$s2 * data$s7)))),
                       s2s8= ifelse(data$s2 * data$s8 == 0, 0, ifelse(data$s2 * data$s8 > 0, sqrt(data$s2 * data$s8), -sqrt(-(data$s2 * data$s8)))),
                       s3s4= ifelse(data$s3 * data$s4 == 0, 0, ifelse(data$s3 * data$s4 > 0, sqrt(data$s3 * data$s4), -sqrt(-(data$s3 * data$s4)))),
                       s3s5= ifelse(data$s3 * data$s5 == 0, 0, ifelse(data$s3 * data$s5 > 0, sqrt(data$s3 * data$s5), -sqrt(-(data$s3 * data$s5)))),
                       s3s6= ifelse(data$s3 * data$s6 == 0, 0, ifelse(data$s3 * data$s6 > 0, sqrt(data$s3 * data$s6), -sqrt(-(data$s3 * data$s6)))),
                       s3s7= ifelse(data$s3 * data$s7 == 0, 0, ifelse(data$s3 * data$s7 > 0, sqrt(data$s3 * data$s7), -sqrt(-(data$s3 * data$s7)))),
                       s3s8= ifelse(data$s3 * data$s8 == 0, 0, ifelse(data$s3 * data$s8 > 0, sqrt(data$s3 * data$s8), -sqrt(-(data$s3 * data$s8)))),
                       s4s5= ifelse(data$s4 * data$s5 == 0, 0, ifelse(data$s4 * data$s5 > 0, sqrt(data$s4 * data$s5), -sqrt(-(data$s4 * data$s5)))),
                       s4s6= ifelse(data$s4 * data$s6 == 0, 0, ifelse(data$s4 * data$s6 > 0, sqrt(data$s4 * data$s6), -sqrt(-(data$s4 * data$s6)))),
                       s4s7= ifelse(data$s4 * data$s7 == 0, 0, ifelse(data$s4 * data$s7 > 0, sqrt(data$s4 * data$s7), -sqrt(-(data$s4 * data$s7)))),
                       s4s8= ifelse(data$s4 * data$s8 == 0, 0, ifelse(data$s4 * data$s8 > 0, sqrt(data$s4 * data$s8), -sqrt(-(data$s4 * data$s8)))),
                       s5s6= ifelse(data$s5 * data$s6 == 0, 0, ifelse(data$s5 * data$s6 > 0, sqrt(data$s5 * data$s6), -sqrt(-(data$s5 * data$s6)))),
                       s5s7= ifelse(data$s5 * data$s7 == 0, 0, ifelse(data$s5 * data$s7 > 0, sqrt(data$s5 * data$s7), -sqrt(-(data$s5 * data$s7)))),
                       s5s8= ifelse(data$s5 * data$s8 == 0, 0, ifelse(data$s5 * data$s8 > 0, sqrt(data$s5 * data$s8), -sqrt(-(data$s5 * data$s8)))),
                       s6s7= ifelse(data$s6 * data$s7 == 0, 0, ifelse(data$s6 * data$s7 > 0, sqrt(data$s6 * data$s7), -sqrt(-(data$s6 * data$s7)))),
                       s7s8= ifelse(data$s7 * data$s8 == 0, 0, ifelse(data$s7 * data$s8 > 0, sqrt(data$s7 * data$s8), -sqrt(-(data$s7 * data$s8)))))


data <- filter(data, class != 2) %>% filter(class != 0)


# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL


# Normalize

# Data partition

set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
training <- data[ind == 1, c(1:8,10:36)]
#training <- apply(training, 2, function(x) x)
test <- data[ind == 2, c(1:8,10:36)]
#test <- apply(test, 2, function(x) x)
trainingtarget <- data[ind==1, 9]
testtarget <- data[ind==2, 9]


# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)[,c(2,4)]
testLabels <- to_categorical(testtarget)[,c(2,4)]
print(testLabels)

# Create sequential model
model <- keras_model_sequential()
model %>%
  layer_dense(units=64, activation = "relu", input_shape = c(NULL,35)) %>%
  layer_dropout(0.2) %>%
  layer_dense(units=32, activation = "relu") %>%
  layer_reshape(target_shape = c(NULL, 32, 1), input_shape = c(NULL, 32)) %>%
  layer_conv_1d(filters=10, kernel_size= 8, padding='same', activation='relu', input_shape = c(NULL, 32, 1)) %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_dropout(0.2) %>%
  layer_flatten() %>%
  layer_dense(units=32, activation = "relu") %>%
  layer_dense(units = 2, activation = "sigmoid")
summary(model)

model <- keras_model_sequential()
model %>%
  layer_dense(units=64, activation = "relu", input_shape = c(NULL, 35)) %>%
  layer_dropout(0.2) %>%
  layer_dense(units=32, activation = "relu") %>%
  layer_dense(units = 2, activation = "sigmoid")
summary(model)

# Compile
model %>%
         compile(loss = "binary_crossentropy",
                 optimizer = "adam",
                 metrics = "accuracy")

# Fit model

history <- model %>%
         fit(training,
             trainLabels,
             epoch = 250,
             batch_size = 64,
             validation_split = 0.1)
plot(history)

# Evaluate model with test data
model1 <- model %>%
         evaluate(test, testLabels)

# Prediction & confusion matrix - test data
prob <- model %>%
         predict_proba(test)

pred <- model %>%
         predict_classes(test)
table1 <- table(Actual = as.factor(testtarget),Predicted = as.factor(ifelse(pred >= 0.5, 3,1)))

table1

# Fine-tune model
