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
originalDataImport = function(folder){
  
  train_org = read.csv(paste(folder, "train.csv", sep = ""), sep = "|", header = TRUE, dec = ".")
  items = read.csv(paste(folder, "items.csv", sep = ""), sep = "|", header = TRUE, dec = ".")
  
  train_org[is.na(train_org)] = 0
  items[is.na(items)] = "MISSING"
  levels(items$campaignIndex)[1] = "MISSING"
  
  train_org = subset(sqldf("select * from train_org join items on train_org.pid = items.pid"), select = -c(3))
  
  #Remove false predictors
  train_org = subset(train_org, select = -c(pid, lineID, order, click, basket))
  
  #factor data
  train_org.factor = c("adFlag", "availability", "manufacturer", "group", "content", "unit", 
                       "genericProduct", "salesIndex", "category", "campaignIndex")
  
  for(i in train_org.factor){
    train_org[i] = as.factor(train_org[[i]])
  }
  
  return(train_org)
}
prepDataImport = function(folder){
  train = read.csv2(paste(folder), "final_sample.csv")
  
  #Remove false predictors
  train = subset(train, select = -c(lineID, pid, order, quantity))
  
  #MISSING VALUE HANDLING
  train$relativediff_comp_price[is.infinite((train$relativediff_comp_price))] = 0
  train$relativediff_rrp_comp[is.infinite((train$relativediff_rrp_comp))] = 0
  levels(train$campaignIndex)[1] = "MISSING"
  
  train.factor_fix = c("adFlag", "availability", "genericProduct", "salesIndex", 
                       "campaignIndex", "weekdays", "missingCompetitorPrice", "ABC_rev", "ABC_dem")
  
  train.factor_rem = c("manufacturer", "category", "contentContainsX", "unitClean")
  
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

#DUMMY CODING
for(i in c(1:length(sets))){
  sets[[i]] = createDummyFeatures(sets[[i]], target = ("revenue"))
}


#Sample selection
#Get uniform sample
uniformSampling = function(listSets, size){
  row_sample = sample(rownames(listSets[[1]]), size)
  
  for(i in c(1:length(listSets))){
    listSets[[i]] = listSets[[i]][row_sample,]
  }
  return(listSets)
}
sets = uniformSampling(sets, 20000)

#MODEL PREPARATION
rdesc = makeResampleDesc("CV", iters = 3)
#learner selection
learners = list(makeLearner("regr.lm"),
                makeFeatSelWrapper("regr.glm", resampling = rdesc, control = makeFeatSelControlGA(maxit = 20, mutation.rate = 0.1)))

#parameter_sets if necessary
#getParamSet("classif.penalized.lasso")
#lasso.learner = makeLearner("regr.penalized.lasso")
#lasso.learner = setHyperPars(lasso.learner, lambda1 = 20, trace = TRUE)
#lasso.wrapper = makeFeatSelWrapper(lasso.learner, resampling = rdesc, control = makeFeatSelControlGA(maxit = 20, mutation.rate = 0.1))
#learners = append(learners, list(lasso.wrapper))

#make tasks
makeTasks = function(listSets, target){
  tasks = list()
  
  for(i in c(1:length(listSets))){
    tasks = append(tasks, list(makeRegrTask(id = as.character(i), data = listSets[[i]], target = target)))
  }
  
  return(tasks)
}
tasks = makeTasks(sets, "revenue")

#RESAMPLING / TESTING DECSISION
rdesc2 = makeResampleDesc("CV", iters = 5)

#EVALUATION
#parralization
#parallelStartBatchJobs()
#parallelStartSocket(4, level = "mlr.resample")
#Setting up a benchmarking experiment

bmr = benchmark(learners, tasks, rdesc2, list(rmse))

#parallelStop()
write.csv(bmr, "test_regr.csv")

#Final evaluation of the models
#MSE
