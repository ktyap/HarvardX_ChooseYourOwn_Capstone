# ------------------------- MovieLens Capstone ---------------------------------
# Yap Kim Thow | 8th Jan 2021

#### SECTION 1: INTRODUCTION AND DATA EXPLORATION
# ------------------- Load packages --------------------------------------------
# load (and install if required) packages used in this project
if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(PerformanceAnalytics)) install.packages("PerformanceAnalytics"); 
library(PerformanceAnalytics) #for correlation plot
if(!require(e1071)) install.packages("e1071"); library(e1071)
if(!require(rattle)) install.packages("rattle"); library(rattle) #for rpart plot
#set number of significant digits
options(digits = 4)

# -------------------------- Data Exploration ----------------------------------
#read data from URL of my Github
dat <- read_csv("https://github.com/ktyap/HarvardX_ChooseYourOwn_Capstone/raw/master/column_3C_weka.csv")
#basic data structure examination
head(dat)
str(dat)
#check for NAs
isNA <- apply(dat, 2, function(x){
  any(is.na(x))
})
isNA
#number of distinct labels in "class"
levels(factor(dat$class))

#mutate class variable to factor
dat <- dat %>% mutate(class = factor(class))
#count of observations in the 3 classes
dat %>% ggplot(aes(x = class)) + geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) + 
  ggtitle("Number of observations by Class") + 
  xlab("Class") + ylab("Count")

#correlation plot of variables
chart.Correlation(dat[,-7])

#Boxplot for "pelvic_incidence"
dat %>% ggplot(aes(x = class, y = pelvic_incidence, fill = class)) + geom_boxplot() +
  ggtitle("Class by Pelvic Incidence") + xlab("Class") + 
  ylab("Pelvic Incidence")
#Boxplot for "pelvic_tilt"
dat %>% ggplot(aes(x = class, y = pelvic_tilt, fill = class)) + geom_boxplot() +
  ggtitle("Class by Pelvic Tilt") + xlab("Class") + ylab("Pelvic Tilt")
#Boxplot for "lumbar_loardosis_angle"
dat %>% ggplot(aes(x = class, y = lumbar_lordosis_angle, fill = class)) + 
  geom_boxplot() +
  ggtitle("Class by Lumbar Lordosis Angle") + xlab("Class") + 
  ylab("Lumbar Lordosis Angle")
#Boxplot for "sacral_slope"
dat %>% ggplot(aes(x = class, y = sacral_slope, fill = class)) + geom_boxplot() +
  ggtitle("Class by Sacral Slope") + xlab("Class") + ylab("Sacral Slope")
#Boxplot for "pelvic_radius"
dat %>% ggplot(aes(x = class, y = pelvic_radius, fill = class)) + geom_boxplot() +
  ggtitle("Class by Pelvic Radius") + xlab("Class") + ylab("Pelvic Radius")
#Boxplot for "degree_spondylolishesis"
dat %>% ggplot(aes(x = class, y = degree_spondylolisthesis, fill = class)) + 
  geom_boxplot() +
  ggtitle("Class by Degree Spondylolisthesis") + xlab("Class") + 
  ylab("Degree Spondylolisthesis")

#examine distribution of Hernia and Normal classes
dat %>% filter(class == "Hernia" | class == "Normal") %>%
          ggplot(aes(x = lumbar_lordosis_angle, y = pelvic_radius, shape = class,
          color = class)) + geom_point()

# ------------------------- Data Partitioning ----------------------------------
set.seed(5, sample.kind="Rounding")
#80/20 split of training and test sets
index <- createDataPartition(y = dat$class, times = 1, p = 0.8, list = FALSE)
train <- dat[index,]
test <- dat[-index,]

#### SECTION 2: METHODS/ANALYSIS

#trainControl for model training (10-fold cross-validation)
trainControl <- trainControl(method = "cv", number = 10, savePredictions = "all")

# ------------------------- Classification Tree Model --------------------------
set.seed(5, sample.kind="Rounding")
#classificaiton tree using rpart
train_rpart <- train(class ~ ., method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.2, by = 0.01)),
                     trControl = trainControl,
                     data = train)

#plot accuracy based on cp values
plot(train_rpart)

#first 5 results of cp values
train_rpart$results[1:5,]

#confusion matrix for cross-validated rpart model
confusionMatrix(train_rpart)

#variable importance based on cross-validated rpart
varImp(train_rpart)

#plot decision tree based on rpart
fancyRpartPlot(train_rpart$finalModel, main = "Decision Tree", sub = "")

# ------------------------ Random Forest Model ---------------------------------
set.seed(5, sample.kind="Rounding")
#random forest using rf package
train_rf <- train(class ~ ., method = "rf",
                  ntree = 1000, tuneGrid = data.frame(mtry = seq(2, 5)),
                  trControl = trainControl,
                  data = train)

#plot accuracies (10-fold cross-validation) for mtry
plot(train_rf)

#confusion matrix for cross-validated rf model
confusionMatrix(train_rf)

#variable importance based on cross-validated rf
varImp(train_rf)

# ------------------------ Support Vector Machine Model ------------------------
set.seed(5, sample.kind="Rounding")
#svm using svmLinear
train_svm <- train(class ~ ., method = "svmLinear",
                   preProcess = c("center", "scale"), #centering and scaling
                   trControl = trainControl,
                   #cost parameter
                   tuneGrid = data.frame(C = c(0.1, 1, 10, 100, 1000, 10000)),
                   data = train)

#plot accuracies (10-fold cross-validation) for C
train_svm %>% ggplot(aes(x = C, y = Accuracy)) + scale_x_log10() +
  ggtitle("Support Vector Machine")

# ------------------------ Classification Tree Predictions ---------------------
#predictions from rpart
pred_rpart <- predict(train_rpart, test)
#accuracy of rpart prediction against test set
acc_rpart <- confusionMatrix(pred_rpart, test$class)$overall["Accuracy"]
acc_rpart

#keep track of model results
acc_results <- tibble(method = "Classification Tree", 
                    "10-fold CV" = as.character(round(train_rpart$results[1,2],4)),
                    "Test Set" = as.character(round(acc_rpart,4)))

# ------------------------ Random Forest Predictions ---------------------------
#predictions from rf
pred_rf <- predict(train_rf, test)
#accuracy of rf prediction against test set
acc_rf <- confusionMatrix(pred_rf, test$class)$overall["Accuracy"]
acc_rf

#keep track of model results
acc_results <- acc_results %>% bind_rows(tibble(method = "Random Forest",
                    "10-fold CV" = as.character(round(train_rf$results[1,2],4)),
                    "Test Set" = as.character(round(acc_rf,4))))

# ------------------------ Support Vector Machine Predictions ------------------
#predictions from rf
#predictions from SVM
pred_svm <- predict(train_svm, test)
#accuracy of SVM prediction against test set
acc_svm <- confusionMatrix(pred_svm, test$class)$overall["Accuracy"]
acc_svm

#keep track of model results
#keep track of model results
acc_results <- acc_results %>% bind_rows(tibble(method = "Support Vector Machine",
                      "10-fold CV" = as.character(round(train_svm$results[3,2],4)),
                      "Test Set" = as.character(round(acc_svm,4))))
acc_results

#### SECTION 3: CONCLUSIONS
# ------------------------ Results Summary -------------------------------------
#results summary
acc_results
