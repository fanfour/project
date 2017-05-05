rm(list = ls())
gc()
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

sample = read.csv2("01_Data/final_sample.csv")

#DATA IMPORT AND PREPARATION
originalDataImport = function(folder){
  
  train_org = read.csv(paste(folder, "train.csv", sep = ""), sep = "|", header = TRUE, dec = ".")
  items = read.csv(paste(folder, "items.csv", sep = ""), sep = "|", header = TRUE, dec = ".")
  
  train_org[is.na(train_org)] = 0
  items[is.na(items)] = "MISSING"
  levels(items$campaignIndex)[1] = "MISSING"
  
  train_org = subset(sqldf("select * from train_org join items on train_org.pid = items.pid"), select = -c(3))
  
  #Remove false predictors
  train_org = subset(train_org, select = -c(pid, lineID, revenue, click, basket))
  
  #factor data
  train_org.factor = c("adFlag", "availability", "manufacturer", "group", "content", "unit", 
                       "genericProduct", "salesIndex", "category", "campaignIndex", "order")
  
  for(i in train_org.factor){
    train_org[i] = as.factor(train_org[[i]])
  }
  
  return(train_org)
}
prepDataImport = function(folder){
  train = read.csv2(paste(folder, "final_sample.csv", sep = ""))
  
  names(train)[names(train)=="order_Clean"] = "order"
  
  #Remove false predictors
  train = subset(train, select = -c(X, lineID, day_Clean, revenue_Clean, content_Clean))
  
  #WEEKDAYS ARE MISSING
  train.factor_fix = c("adFlag_Clean", "availability_Clean", "order", "pharmForm_Clean", 
                       "genericProduct_Clean", "salesIndex_Clean", "missingCompetitorPrice")
  
  train.factor_rem = c("manufacturer_Clean", "group_Clean", "unit_Clean", "category_Clean", "campaignIndex_Clean")
  
  train.factor = c(train.factor_fix, train.factor_rem)
  
  for(i in train.factor){
    train[i] = as.factor(train[[i]])
  }
  
  return(train)
}
sets = list(originalDataImport("01_Data/"), prepDataImport("01_Data/"))

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

#Sample selection
#Get uniform sample
uniformSampling = function(listSets, size){
  row_sample = sample(rownames(listSets[[1]]), size)
  
  for(i in c(1:length(listSets))){
    row_sample = sample(rownames(listSets[[i]]), size)
    
    listSets[[i]] = listSets[[i]][row_sample,]
  }
  return(listSets)
}
sets = uniformSampling(sets, 20000)

#MODEL PREPARATION
rdesc = makeResampleDesc("CV", iters = 3)
#learner selection
learners = list(makeLearner("classif.OneR"),
                makeFeatSelWrapper("classif.C50", resampling = rdesc, control = makeFeatSelControlGA(maxit = 10, mutation.rate = 0.1)))

#makeLearner("classif.boosting")
#makeFeatSelWrapper("classif.randomForest", resampling = rdesc, control = makeFeatSelControlGA(maxit = 10, mutation.rate = 0.1))

#parameter_sets if necessary
#getParamSet("classif.penalized.lasso")
lasso.learner = makeLearner("classif.penalized.lasso")
lasso.learner = setHyperPars(lasso.learner, lambda1 = 10, trace = TRUE)
lasso.wrapper = makeFeatSelWrapper(lasso.learner, resampling = rdesc, control = makeFeatSelControlGA(maxit = 10, mutation.rate = 0.1))

learners = append(learners, list(lasso.wrapper))

#make tasks
makeTasks = function(listSets, target){
  tasks = list()
  
  for(i in c(1:length(listSets))){
    tasks = append(tasks, list(makeClassifTask(id = as.character(i), data = listSets[[i]], target = target)))
  }
  
  return(tasks)
}
tasks = makeTasks(sets, "order")

#RESAMPLING / TESTING DECSISION
rdesc2 = makeResampleDesc("CV", iters = 5)

#EVALUATION
#parralization
#parallelStartBatchJobs()
parallelStartSocket(2, level = "mlr.resample")
#Setting up a benchmarking experiment

bmr = benchmark(learners, tasks, rdesc2, list(acc, fpr, fnr))

parallelStop()
write.csv(bmr, "test.csv")
#Final evaluation of the models
#MSE

#Get confusion matrix of all learner
library(SDMTools)
confusion.matrix(getBMRPredictions(bmr, learner.ids = "classif.OneR", as.df = TRUE)$truth, getBMRPredictions(bmr, learner.ids = "classif.OneR", as.df = TRUE)$response)

prediction = getBMRPredictions(bmr, learner.ids = "classif.OneR", as.df = TRUE)
calculateConfusionMatrix(prediction)
table(as.numeric(getBMRPredictions(bmr, learner.ids = "classif.OneR", as.df = TRUE)$truth))
table(as.numeric(getBMRPredictions(bmr, learner.ids = "classif.OneR", as.df = TRUE)$response))

sqldf("select CAST(t1.tr AS float) / CAST(count(*) AS float) as acc  from prediction, (select count(*) as tr from prediction where truth = response) as t1")
