rm(list=ls())
library(mlr)
library(sqldf)
#library(FSelector)
library(party)
#install.packages("party")

set.seed(42)

#Get all colnames except
#setwd("C:/SEMINAR/DMC_GITHUB/project/DMC")

prepDataImport = function(){
  train = read.csv("C:/SEMINAR/DATA/SAMPLE_DATA/mergedTrain.csv")
  
  #Remove complex attributes
  
  #train = head(subset(train, select = -c(lineID, pid_Clean, click_Clean, basket_Clean)),2000)
  train = subset(train, select = -c(lineID, pid_Clean, click_Clean, basket_Clean))
  #mean_basket,competitorPrice_Clean,rrp_Clean, minPrice,maxPrice, avgPrice,diff_comp_price, 
  #diff_rrp_comp,diff_rrp_price,diff_min_price,diff_max_price,diff_avg_price))
  
  #Remove false predictors
  train = subset(train, select = -c(quantity, order_Clean))
  
  #Remove redundant features
  train = subset(train, select = -c(maxPrice, avgPrice, minPrice, rrp_Clean, competitorPrice_Clean))
  
  #Remove perfect collinear attributes
  
  names(train)[names(train)=="order_Clean"] = "order"
  names(train)[names(train)=="price_Clean"] = "price"
  
  #WEEKDAYS ARE MISSING
  train.factor= c("adFlag_Clean", "availability_Clean", "pharmForm_Clean", 
                       "genericProduct_Clean", "salesIndex_Clean", "missingCompetitorPrice", "weekdays",
                       "manufacturer_Clean", "group_Clean", "content_Clean", "unit_Clean", "category_Clean", "campaignIndex_Clean")
  

  for(i in train.factor){
    train[i] = as.factor(train[[i]])
  }
  
  return(train)
}

data = prepDataImport()

data.test = sqldf("select * from data where day_Clean > 75")
data.train = sqldf("select * from data where day_Clean <= 75")
rm(data)

train.set = sample(rownames(data.train), 1000)
#train.set = rownames(data.train[[1]])
data.train = data.train[train.set,]
data = rbind(data.train, data.test)

sets = list(data)
rdec  = makeResampleDesc("Holdout", split = length(data.train[[1]])/length(data[[1]]))
rm(data, train.set, data.train, data.test)

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

#lrn1 = makeLearner("regr.randomForest")

#lrn2 = makeLearner("regr.randomForestSRC"), fw.method = "chi.squared", fw.threshold = 1, fw.mandatory.feat = col.keep)
lrn2 = makeFilterWrapper(makeLearner("regr.cforest"), fw.method = "cforest.importance", fw.abs = 10)
lrn3 = makeFeatSelWrapper(makeLearner("regr.cforest"),rdec , control = makeFeatSelControlSequential(method = "sfs"))
?makeFeatSelWrapper
stack = makeStackedLearner(list(lrn2, lrn3), predict.type = "response", method = "hill.climb")

learners = list(lrn2,
                lrn3)
                #,stack)

tasks = list()

for(i in sets){
  task = makeRegrTask(data = i, target = "revenue_Clean")
  tasks = append(tasks, list(task))
}

test = train(lrn3, task)


ps = makeParamSet(
  makeDiscreteParam("ntree", values = c(50,100)),
  makeDiscreteParam("mtry", values = c(3,5,10))
  #,makeDiscreteParam("maxdepth",values = c(3,4,5,6,7,8,9,10))                  
)

cntr = makeTuneControlGrid()

tuneParams(learners, tasks[[1]], rdec,  measures = rmse, par.set = ps, control = cntr)

#bmr = benchmark(learners, tasks, rdesc, measures = rmse)

