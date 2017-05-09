library(mlr)

create_costs = function(){
  #get sample
  sample = read.csv("C:/Users/Manuel/OneDrive/Project_Datascience/R/DMC_2017_Seminar/01_Data/mergedTrain_sample.csv")
  
  cost_y = sample$price_Clean*sample$avg_quantity*(1-sample$mean_order)
  cost_n = sample$price_Clean*sample$avg_quantity*sample$mean_order
  costs = data.frame(cost_n, cost_y)
  
  colnames(costs) = levels(as.factor(sample$order_Clean))
  rownames(costs) = sample$lineID_Clean
  
  return(costs)
}

#costs for yes are higher -> false positive rate gets punshed
cost = create_costs()

df = read.csv("C:/Users/Manuel/OneDrive/Project_Datascience/R/DMC_2017_Seminar/01_Data/mergedTrain_sample.csv")

#Remove complex attributes
df = subset(df, select = -c(lineID, pid_Clean, click_Clean, basket_Clean, group_Clean, category_Clean, pharmForm_Clean, manufacturer_Clean))
#Remove perfect collinear attributes
df = subset(df, select = -c(mean_basket, competitorPrice_Clean, rrp_Clean, minPrice, maxPrice, avgPrice, diff_comp_price, diff_rrp_comp, diff_rrp_price, diff_min_price, diff_max_price, diff_avg_price))
sets = list(df)

learners = list(makeCostSensWeightedPairsWrapper(makeLearner("classif.C50")), makeCostSensWeightedPairsWrapper(makeLearner("classif.multinom")))
reg_learners = list(makeLearner("regr.lm"))

#make tasks
makeCostSensTasks = function(listSets, target){
  tasks = list()
  
  for(i in c(1:length(listSets))){
    data = subset(listSets[[i]], select = -c(quantity, order_Clean, revenue_Clean))
    tasks = append(tasks, list(makeCostSensTask(id = as.character(i), data = data, cost = cost)))
  }
  
  return(tasks)
}
makeRegrTasks = function(listSets, target){
  regTasks = list()
  
  for(i in c(1:length(listSets))){
    
    #REGRESION OVER QUANTITY
    data = createDummyFeatures(subset(listSets[[i]], select = -c(order_Clean, revenue_Clean)))
    newTask = makeRegrTask(id = as.character(i), data = data, target = target)
    regTasks = append(regTasks, list(newTask))
  }
  
  return(regTasks)
}



training_and_predicting = function(sets, learnerClass, learnerRegr, TrainTestRatio){
  #Remove quantity for prediction
  class_tasks = makeCostSensTasks(sets)
  
  
  list_pred = list()
  for(i in c(1:length(class_tasks))){
    #Get the train set
    n = length(sets[[i]][[1]])
    train_class = head(c(1:n), ceiling(TrainTestRatio*n))
    test_class = tail(c(1:n), ceiling((1-TrainTestRatio)*n))
    
    
    for(j in learnerClass){
      model = train(j, class_tasks[[i]], subset = train_class)
      predictions = predict(model, class_tasks[[i]], subset = test_class)
      
      pred_sum = data.frame(class_pred = predictions$data$response)
      pred_sum["class_pred"] = as.numeric(pred_sum$class_pred)-1
      pred_sum["id"] = predictions$data$id
      pred_sum["classifier"] = j$id
      pred_sum["Task"] = i
      pred_sum["price"] = head(sets[[i]]$price_Clean, ceiling((1-TrainTestRatio)*n))
      pred_sum["revenue_Clean"] = head(sets[[i]]$revenue_Clean, ceiling((1-TrainTestRatio)*n))
      list_pred = append(list_pred, list(pred_sum))
    }
  }
  
  
  regTasks = makeRegrTasks(sets, "quantity")
  
  list_pred_reg = list()
  for(i in c(1:length(regTasks))){
    
    train_reg = as.numeric(rownames(head(sets[[i]][which(sets[[i]]$quantity > 0),], TrainTestRatio*n)))
    test_reg = as.numeric(tail(rownames(sets[[i]]), (1-TrainTestRatio)*n))
    
    
    for(j in learnerRegr){
      model = train(j, regTasks[[i]], subset = train_reg)
      predictions = predict(model, regTasks[[i]], subset = test_reg)
      
      pred_sum = data.frame(reg_pred = predictions$data$response)
      pred_sum["id"] = predictions$data$id
      pred_sum["regression_alg"] = j$id
      list_pred_reg = append(list_pred_reg, list(pred_sum))
    }
  }
  
  #combine the results
  final_results = list()
  
  
  for(i in c(1:length(list_pred))){
    for(j in c(1:length(list_pred_reg)))
      final_results = append(final_results, list(merge(list_pred[[i]], list_pred_reg[[j]], by = "id")))
  }
  
  return(final_results)
}



#TRAINING AND PREDICTING ALL
#parallelStartBatchJobs()
parallelStartSocket(2, level = "mlr.resample")
final_res = training_and_predicting(sets, learners, reg_learners, 0.7)
parallelStop()

summary(final_res[[1]]$class_pred)
table(final_res[[1]]$class_pred)


#FINAL PERFORMANCE EVALUATION
revenue_calculation = function(comb_preds){
  ret = list()
  for(i in comb_preds){
    tmp = i
    tmp["revenuePrediction"] = tmp$class_pred*tmp$reg_pred*tmp$price
    ret = append(ret, list(tmp))
  }
  
  return(ret)
}
rev_preds = revenue_calculation(final_res)

result_viz = function(rev_preds){
  par(mfrow=c(length(rev_preds),1))
  
  for(i in rev_preds){
    test = i[order(i$revenue_Clean),]
    plt = plot(c(1:length(test[[1]])), test$revenue_Clean, type = "l", xlab = paste(test$classifier, "+", test$regression_alg, "+", test$Task))
    print(plt)
    ln = lines(c(1:length(test[[1]])), test$revenuePrediction, col = "red")
    print(ln)
  }
}

result_viz(rev_preds)


rmse_own = function(actual, predicted){
  error <- actual - predicted
  rmse = sqrt(mean(error^2))
  
  return(rmse)
}
rmse_own(rev_preds[[1]]$revenue_Clean, integer(length(rev_preds[[1]]$revenue_Clean)))

for(i in rev_preds){
  print(rmse_own(i$revenue_Clean, i$revenuePrediction))
}

summary(rev_preds[[1]]$revenuePrediction)