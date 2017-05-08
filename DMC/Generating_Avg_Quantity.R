library(sqldf)
#getwd()
#setwd("~/SEMINAR/DMC_GITHUB/project/DMC/")

joined_train_org_sample = read.csv("C:/SEMINAR/DMC_GITHUB/project/DMC/joined_train_org_sample.csv", sep=";", header=TRUE, dec=".")

mergedTrain_sample = read.csv("C:/SEMINAR/DMC_GITHUB/project/DMC/mergedTrain_sample.csv", sep=",", header=TRUE, dec=".")

view(mergedTrain_sample);

head(joined_train_org_sample, n=4)

original_train = read.csv("C:/SEMINAR/ORIGINAL_DATA/train.csv", sep= "|", header = TRUE, dec=".")

average_quantity = sqldf("select pid,avg(revenue/price) as avg_quantity from original_train group by pid")

names(mergedTrain_sample)[8] <- "pid"

mergedTrainQ_sample = merge(mergedTrain_sample,average_quantity,by="pid")

write.csv(mergedTrain_sample,"C:/SEMINAR/DMC_GITHUB/project/DMC/Training_Sample.csv")

write.csv(average_quantity,"C:/SEMINAR/DMC_GITHUB/project/DMC/avg_quantity.csv")
