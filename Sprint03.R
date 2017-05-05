rm(list=ls())
setwd("~/Documents/Master_TUM/2017_SS/Seminar Data Mining/Sprint03")   

sample <- read.table("mergedTrain_Sample.csv", sep=",", header=TRUE, dec=".")  
head(sample, n=4)

str(sample)

sample$manufacturer_Clean = as.factor(sample$manufacturer_Clean)
sample$availability_Clean = as.factor(sample$availability_Clean)
sample$click_Clean = as.factor(sample$click_Clean)
sample$basket_Clean = as.factor(sample$basket_Clean)
sample$content_Clean = as.factor(sample$content_Clean)
sample$salesIndex_Clean = as.factor(sample$salesIndex_Clean)
sample$campaignIndex_Clean = as.factor(sample$campaignIndex_Clean)
sample$missingCompetitorPrice = as.factor(sample$missingCompetitorPrice)
sample$weekdays = as.factor(sample$weekdays)
sample$order_Clean = as.factor(sample$order_Clean)

sample = subset(sample, select = -c(lineID, quantity))


library(mlr)

# Feature Selection
# Calculate weights for the attributes using Info Gain and Gain Ratio
task = makeClassifTask(data=sample, target="order_Clean")
filter = generateFilterValuesData(task, method = c("information.gain","chi.squared"))
ranking = filter$data[with(filter$data, order(-information.gain)),]
row.names(ranking) = c(1:length(filter$data[[1]]))
ranking

# Select the most important attributes based on Gain Ratio
#most_important_attributes <- cutoff.k(ranking, 10)
#most_important_attributes
#formula_with_most_important_attributes <- as.simple.formula(most_important_attributes, "order_Clean")
#formula_with_most_important_attributes


#####
library(mlr)

## 1) Define the task
## Specify the type of analysis (e.g. classification) and provide data and response variable
task = makeRegrTask(data = sample, target = "revenue_Clean")

## 2) Define the learner
## Choose a specific algorithm (e.g. linear discriminant analysis)
lrn = makeLearner("regr.lm")

n = nrow(sample)
train.set = sample(n, size = 2/3*n)
test.set = setdiff(1:n, train.set)

## 3) Fit the model
## Train the learner on the task using a random subset of the data as training set
model = train(lrn, task, subset = train.set)

## 4) Make predictions
## Predict values of the response variable for new observations by the trained model
## using the other part of the data as test set
pred = predict(model, task = task, subset = test.set)

## 5) Evaluate the learner
## Calculate the mean misclassification error and accuracy
performance(pred, measures = list(mmce, acc))
#> mmce  acc 
#> 0.04 0.96

####
# Error
error <- actual - predicted

# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Example of invocation of functions
rmse(error)

