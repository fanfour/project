rm(list=ls())
library(mlr)
set.seed(42)

#Get all colnames except


prepDataImport = function(){
  train = read.csv("mergedTrain_sample.csv")
  
  #Remove complex attributes
  train = subset(train, select = -c(lineID, pid_Clean, click_Clean, basket_Clean))
  #Remove false predictors
  train = subset(train, select = -c(quantity, order_Clean))
  
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

data = prepDataImport()
sets = list(list(data))

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


col.rem = list("mean_basket", 
       "competitorPrice_Clean", 
       "rrp_Clean", "minPrice", 
       "maxPrice", "avgPrice", 
       "diff_comp_price", 
       "diff_rrp_comp", 
       "diff_rrp_price", 
       "diff_min_price", 
       "diff_max_price", 
       "diff_avg_price")

col.keep = colnames(data)[!colnames(sets[[]]) %in% col.rem]

lrn1 = makeLearner("regr.lm")
lrn2 = makeFilterWrapper(lrn1, method="information.gain", fw.threshold = 1, fw.mandatory.feat = col.keep)

stack = makeStackedLearner(list(lrn1, lrn2), predict.type = "response", method = "hill.climb")

learners = list(lrn1,
                lrn2,
                stack)

task = makeRegrTask(data = data, target = "revenue_Clean")
rdesc = makeResampleDesc("CV", iters = 3)

bmr = benchmark(learners, task, rdesc, measures = rmse)

