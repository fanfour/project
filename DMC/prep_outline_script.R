rm(list=ls())
library(mlr)
library(party)
set.seed(42)

#Get all colnames except


prepDataImport = function(){
  train = head(read.csv("mergedTrain_sampleLarge.csv"), 100000)
  
  #Remove complex attributes
  train = subset(train, select = -c(X, lineID, pid_Clean, click_Clean, basket_Clean))
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
sets = list(data)

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


col.rem = c("mean_basket", 
            "competitorPrice_Clean", 
            "rrp_Clean", "minPrice", 
            "maxPrice", "avgPrice", 
            "diff_comp_price", 
            "diff_rrp_comp", 
            "diff_rrp_price", 
            "diff_min_price", 
            "diff_max_price", 
            "diff_avg_price")

target = "revenue_Clean"

col.rem = c(col.rem, target)

for(i in c(1:length(sets))){
  sets[[i]] = createDummyFeatures(sets[[i]], target = ("revenue_Clean"))
}

#standardize the features
sets = list(normalizeFeatures(sets[[1]], method = "standardize", target = "revenue_Clean"))

remove_collinear_attributes = function(sets){
  ret = list()
  
  for(i in sets){
    X <- as.matrix(i)
    
    qr.X <- qr(X, tol=1e-9, LAPACK = FALSE)
    (rnkX <- qr.X$rank)  ## 4 (number of non-collinear columns)
    (keep <- qr.X$pivot[seq_len(rnkX)])
    ## 1 2 4 5 
    X2 <- X[,keep]
    ret = append(ret, list(as.data.frame(X2)))
    
  }
  
  return(ret)
}
sets = remove_collinear_attributes(sets)

col.keep = colnames(sets[[1]])[!colnames(sets[[1]]) %in% col.rem]

# lrn1 = makeLearner("regr.lm")
# lrn2 = makeFilterWrapper("regr.lm", fw.method = "chi.squared", fw.threshold = 1, fw.mandatory.feat = col.keep)

# lrn1 = makeLearner("regr.ctree")
# lrn2 = makeFilterWrapper("regr.ctree", fw.method = "chi.squared", fw.threshold = 1, fw.mandatory.feat = col.keep)

#use the standard configuration with reduced number of trees
#controls = cforest_unbiased(ntree = 50)

# lrn1 = makeLearner("regr.bcart")
# lrn2 = makeFilterWrapper(makeLearner("regr.cart"), fw.method = "chi.squared", fw.threshold = 1, fw.mandatory.feat = col.keep)

lrn1 = makeLearner("regr.blm")
lrn2 = makeFilterWrapper(makeLearner("regr.blm"), fw.method = "chi.squared", fw.threshold = 1, fw.mandatory.feat = col.keep)

stack = makeStackedLearner(list(lrn1, lrn2), predict.type = "response", method = "hill.climb")

learners = list(lrn1,
                lrn2)


task = makeRegrTask(data = sets[[1]], target = "revenue_Clean")


test = train(lrn1, task)
rdesc = makeResampleDesc("CV", iters = 3)

bmr = benchmark(learners, task, rdesc, measures = rmse)

