rm(list=ls())
library(mlr)
library(parallelMap)
set.seed(42)

#Get all colnames except


prepDataImport = function(){
  train = read.csv("mergedTrain_sample.csv")
  
  #Remove complex attributes
  train = subset(train, select = -c(lineID, pid_Clean, click_Clean, basket_Clean))
  #Remove false predictors
  train = subset(train, select = -c(quantity, order_Clean))
  #Remove perfectly colinear
  train = subset(train, select = -c(mean_basket, competitorPrice_Clean, rrp_Clean, minPrice, maxPrice, avgPrice, diff_comp_price, diff_rrp_comp, diff_rrp_price, diff_min_price, diff_max_price, diff_avg_price))
  
  
  #Remove perfect collinear attributes
  
  names(train)[names(train)=="order_Clean"] = "order"
  names(train)[names(train)=="price_Clean"] = "price"
  
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

sets = list(prepDataImport())

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

ps = makeParamSet(
  makeDiscreteParam("ntree", values = c(25, 50, 100, 500)),
  makeDiscreteParam("bootstrap", values = c("by.node")),
  makeDiscreteParam("nodesize", values = c(3,5,10))
)
ctrl = makeTuneControlGrid()

lrn = makeLearner("regr.randomForestSRC")

task = makeRegrTask(data = sets[[1]], target = "revenue_Clean")
rdec = makeResampleDesc("CV", iters = 3)

#parallelStartMPI(logging = TRUE)
parallelStartSocket(logging = TRUE)
res = tuneParams(lrn, task, rdec, measures = rmse, ps, ctrl, show.info = TRUE)
test = generateHyperParsEffectData(res, partial.dep = TRUE)
test$data
parallelStop()