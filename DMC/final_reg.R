library("parallelMap")
library(mlr)
library(sqldf)
set.seed(42)
options(scipen = 999)


#Installations necessary?
if(FALSE){
  install.packages("penalized")
  install.packages("C50")
  install.packages("randomForest")
  install.packages("adabag")
}

#DATA IMPORT AND PREPARATION
originalDataImport = function(){
  
  train_org = read.csv2("joined_train_org_sample.csv")
  names(train_org)[names(train_org)=="revenue"] = "revenue_Clean"
  
  #Remove false predictors
  train_org = subset(train_org, select = -c(X, quantity, order, pid, lineID))
  
  #factor data
  train_org.factor = c("adFlag", "availability", "manufacturer", "group", "content", "unit", 
                       "genericProduct", "salesIndex", "category", "campaignIndex")
  
  for(i in train_org.factor){
    train_org[i] = as.factor(train_org[[i]])
  }
  
  return(train_org)
}
prepDataImport = function(){
  train = read.csv("mergedTrain_sample.csv")
  
  names(train)[names(train)=="order_Clean"] = "order"
  names(train)[names(train)=="price_Clean"] = "price"
  
  #Remove false predictors
  train = subset(train, select = -c(X, quantity, order, pid_Clean, lineID))
  
  #WEEKDAYS ARE MISSING
  train.factor_fix = c("adFlag_Clean", "availability_Clean", "pharmForm_Clean", 
                       "genericProduct_Clean", "salesIndex_Clean", "missingCompetitorPrice", "weekdays")
  
  train.factor_rem = c("manufacturer_Clean", "group_Clean", "content_Clean", "unit_Clean", "category_Clean", "campaignIndex_Clean")
  
  train.factor = c(train.factor_fix, train.factor_rem)
  
  for(i in train.factor){
    train[i] = as.factor(train[[i]])
  }
  
  return(train)
}
sets = list(originalDataImport(), prepDataImport())

removeIDLikeFactors = function(listSets, maxNrFactors){
  #Remove all factorial attributes with nr(factors) > 50
  for(i in c(1:length(listSets))){
    test = colnames(listSets[[i]][, sapply(listSets[[i]], is.factor)])
    
    rem = c(NULL)
    for(j in test){
      if(length(table(listSets[[i]][j])) > maxNrFactors)
        rem = c(rem, j)
    }
    
    listSets[[i]] = listSets[[i]][, (! colnames(listSets[[i]]) %in% rem)]
  }
  return(listSets)
}
sets = removeIDLikeFactors(sets, 50)

#DUMMY CODING
for(i in c(1:length(sets))){
  sets[[i]] = createDummyFeatures(sets[[i]], target = ("revenue_Clean"))
}

#MODEL PREPARATION
rdesc = makeResampleDesc("Holdout")
#learner selection
lrns = lapply(c("regr.lm", "regr.ctree"), makeLearner)

learners = list(makeLearner("regr.lm"),
                makeLearner("regr.ctree"),
                makeStackedLearner(base.learners = lrns, predict.type = "response", method = "hill.climb"))

#makeStackedLearner(base.learners = lrns, super.learner = "regr.lm", method = "stack.nocv"),

# makeStackedLearner(base.learners = lrns, super.learner = "regr.lm", method = "stack.nocv")
#makeFeatSelWrapper(makeStackedLearner(base.learners = lrns, predict.type = "response", method = "stack.nocv"), resampling = rdesc, control = makeFeatSelControlGA(maxit = 5, mutation.rate = 0.1)))

#make tasks
makeTasks = function(listSets, target){
  tasks = list()
  
  for(i in c(1:length(listSets))){
    tasks = append(tasks, list(makeRegrTask(id = as.character(i), data = listSets[[i]], target = target)))
  }
  
  return(tasks)
}
tasks = makeTasks(sets, "revenue_Clean")

#RESAMPLING / TESTING DECSISION
rdesc2 = makeResampleDesc("CV", iters = 3)

#EVALUATION
#parralization
#parallelStartBatchJobs()
#parallelStartSocket(4, level = "mlr.resample")
#Setting up a benchmarking experiment

bmr = benchmark(learners, tasks, rdesc2, list(rmse))

test = bmr$results$`1`$regr.ctree$pred$data
test = test[order(test$truth),]

par(mfrow = c(1,1))
plot(c(1:length(test[[1]])), test$truth, type = "l")
lines(c(1:length(test[[1]])), test$response, col = "red")

result_viz = function(bmr) {
  par(mfrow = c(length(length(bmr[[1]])*length(bmr[[1]][[1]])), 1))

  for (i in bmr$results$`1`$regr.lm$pred) {
    for(j in i){
      tmp.data = j$pred$data
      tmp.data = dmp.data[order(tmp.data$truth),]
      j$pred$data$truth
    }
    
    plt = plot(
      c(1:length(test[[1]])),
      test$revenue_Clean,
      type = "l",
      xlab = paste(test$classifier, "+", test$regression_alg, "+", test$Task)
    )
    print(plt)
    ln = lines(c(1:length(test[[1]])), test$revenuePrediction, col = "red")
    print(ln)
  }
}

result_viz(rev_preds)

#parallelStop()
write.csv(bmr, "test_regr.csv")

#Final evaluation of the models
#MSE
