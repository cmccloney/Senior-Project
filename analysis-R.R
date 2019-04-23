require(ggplot2)
library(car)
#read in datasets
A_Bayes <- read.csv("A_Bayes.csv")
A_Cat <- read.csv("A_Cat.csv")
A_Neighbors <- read.csv("A_Neighbors.csv")
A_NN <- read.csv("A_NN.csv")
A_RF <- read.csv("A_RF.csv")
M_Bayes <- read.csv("M_Bayes.csv")
M_Cat <- read.csv("M_Cat.csv")
M_Neighbors <- read.csv("M_Neighbors.csv")
M_NN <- read.csv("M_NN.csv")
M_RF <- read.csv("M_RF.csv")
PC_Bayes <- read.csv("PC_Bayes.csv")
PC_Cat <- read.csv("PC_Cat.csv")
PC_Neighbors <- read.csv("PC_Neighbors.csv")
PC_NN <- read.csv("PC_NN.csv")
PC_RF <- read.csv("PC_RF.csv")
#means
acc.mean = data.frame(City = c(rep("Austin",5), rep("Mesa",5), rep("Palm Coast",5)),
                      Algorithm = c("Bayes","Catboost","Neighbors","Neural Network","Random Forest"),
                      Accuracy_Mean = c(mean(A_Bayes$Accuracy), mean(A_Cat$Accuracy), mean(A_Neighbors$Accuracy), mean(A_NN$Accuracy), mean(A_RF$Accuracy),
                                mean(M_Bayes$Accuracy), mean(M_Cat$Accuracy), mean(M_Neighbors$Accuracy), mean(M_NN$Accuracy), mean(M_RF$Accuracy),
                                mean(PC_Bayes$Accuracy), mean(PC_Cat$Accuracy), mean(PC_Neighbors$Accuracy), mean(PC_NN$Accuracy), mean(PC_RF$Accuracy)))
pre.mean = data.frame(City = c(rep("Austin",5), rep("Mesa",5), rep("Palm Coast",5)),
                      Algorithm = c("Bayes","Catboost","Neighbors","Neural Network","Random Forest"),
                      Precision_Mean = c(mean(A_Bayes$Precision), mean(A_Cat$Precision), mean(A_Neighbors$Precision), mean(A_NN$Precision), mean(A_RF$Precision),
                                        mean(M_Bayes$Precision), mean(M_Cat$Precision), mean(M_Neighbors$Precision), mean(M_NN$Precision), mean(M_RF$Precision),
                                        mean(PC_Bayes$Precision), mean(PC_Cat$Precision), mean(PC_Neighbors$Precision), mean(PC_NN$Precision), mean(PC_RF$Precision)))
rec.mean = data.frame(City = c(rep("Austin",5), rep("Mesa",5), rep("Palm Coast",5)),
                      Algorithm = c("Bayes","Catboost","Neighbors","Neural Network","Random Forest"),
                      Recall_Mean = c(mean(A_Bayes$Recall), mean(A_Cat$Recall), mean(A_Neighbors$Recall), mean(A_NN$Recall), mean(A_RF$Recall),
                                        mean(M_Bayes$Recall), mean(M_Cat$Recall), mean(M_Neighbors$Recall), mean(M_NN$Recall), mean(M_RF$Recall),
                                        mean(PC_Bayes$Recall), mean(PC_Cat$Recall), mean(PC_Neighbors$Recall), mean(PC_NN$Recall), mean(PC_RF$Recall)))

#create new data frame for each metric
accuracy.df = data.frame(City = c(rep("Austin",150), rep("Mesa",150), rep("Palm Coast",150)),
                    Algorithm = c(rep(c(rep("Bayes",30), rep("Catboost",30), rep("Neighbors",30), rep("Neural Network",30), rep("Random Forest",30)),3)),
                    Accuracy = c(A_Bayes$Accuracy, A_Cat$Accuracy[seq(1,30)], A_Neighbors$Accuracy, A_NN$Accuracy, A_RF$Accuracy[seq(1,30)],
                              M_Bayes$Accuracy, M_Cat$Accuracy[seq(1,30)], M_Neighbors$Accuracy, M_NN$Accuracy, M_RF$Accuracy[seq(1,30)],
                              PC_Bayes$Accuracy, PC_Cat$Accuracy[seq(1,30)], PC_Neighbors$Accuracy, PC_NN$Accuracy, PC_RF$Accuracy[seq(1,30)]))
precision.df = data.frame(City = c(rep("Austin",150), rep("Mesa",150), rep("Palm Coast",150)),
                          Algorithm = c(rep(c(rep("Bayes",30), rep("Catboost",30), rep("Neighbors",30), rep("Neural Network",30), rep("Random Forest",30)),3)),
                          Precision = c(A_Bayes$Precision, A_Cat$Precision[seq(1,30)], A_Neighbors$Precision, A_NN$Precision, A_RF$Precision[seq(1,30)],
                                        M_Bayes$Precision, M_Cat$Precision[seq(1,30)], M_Neighbors$Precision, M_NN$Precision, M_RF$Precision[seq(1,30)],
                                        PC_Bayes$Precision, PC_Cat$Precision[seq(1,30)], PC_Neighbors$Precision, PC_NN$Precision, PC_RF$Precision[seq(1,30)]))
recall.df = data.frame(City = c(rep("Austin",150), rep("Mesa",150), rep("Palm Coast",150)),
                       Algorithm = c(rep(c(rep("Bayes",30), rep("Catboost",30), rep("Neighbors",30), rep("Neural Network",30), rep("Random Forest",30)),3)),
                       Recall = c(A_Bayes$Recall, A_Cat$Recall[seq(1,30)], A_Neighbors$Recall, A_NN$Recall, A_RF$Recall[seq(1,30)],
                                M_Bayes$Recall, M_Cat$Recall[seq(1,30)], M_Neighbors$Recall, M_NN$Recall, M_RF$Recall[seq(1,30)],
                                PC_Bayes$Recall, PC_Cat$Recall[seq(1,30)], PC_Neighbors$Recall, PC_NN$Recall, PC_RF$Recall[seq(1,30)]))

#data visualization
ggplot(accuracy.df, aes(Accuracy, Algorithm, colour = City)) + geom_point()
ggplot(precision.df, aes(Precision, Algorithm, colour = City)) + geom_point()
ggplot(recall.df, aes(Recall, Algorithm, colour = City)) + geom_point()
#models
accuracy.model <- aov(Accuracy ~ City*Algorithm, data = accuracy.df)
precision.model <- aov(Precision ~ City*Algorithm, data = precision.df)
recall.model <- aov(Recall ~ City*Algorithm, data = recall.df)
#analysis of overall model
summary(accuracy.model)
summary(precision.model)
summary(recall.model)
#residual plots and assumptions
#1. Homogoneity of variance - Violated
plot(accuracy.model,1)
plot(precision.model,1)
plot(recall.model,1)
leveneTest(Accuracy ~ City*Algorithm, data = accuracy.df)
leveneTest(Precision ~ City*Algorithm, data = precision.df)
leveneTest(Recall ~ City*Algorithm, data = recall.df)
#2. Normality - Violated
plot(accuracy.model,2)
plot(precision.model,2)
plot(recall.model,2)
acc.res <- residuals(object = accuracy.model)
pre.res <- residuals(object = precision.model)
rec.res <- residuals(object = recall.model)
shapiro.test(x = acc.res)
shapiro.test(x = pre.res)
shapiro.test(x = rec.res)
###Because both assumptions are violated, we will instead use bootstrapping,
###as transformations didn't work either (see bottom of code)

library(boot)
set.seed(03182019)
#Accuracy
n <- 30*2 #simulating taking two samples of 30, so times 2
B <- 10000 #number of resamples we'll run
Cities <- c("Austin","Mesa","Palm_Coast")
Algorithms <- c("Bayes","Catboost","Neighbors","Neural_Network","Random_Forest")
veclist <- do.call(paste,expand.grid(Cities,Algorithms)) #get every combo of city and algo
count <- 1
p.values <- rep(-1,105)
for(i in 1:15){
  temp <- strsplit(veclist[i], " ")[[1]]
  name1 <- gsub("_", " ", temp[1]) #City1
  name2 <- gsub("_", " ", temp[2]) #Algo1
  for(j in (i+1):15){
    temp2 <- strsplit(veclist[j], " ")[[1]]
    name3 <- gsub("_", " ", temp2[1]) #City2
    name4 <- gsub("_", " ", temp2[2]) #Algo2
    if( i != 15){ #prevent last two comparisons, so theres ((15-1)*15) / 2 = 105 total comparisons
      variable1 <- accuracy.df$Accuracy[accuracy.df$City == name1 & accuracy.df$Algorithm == name2]
      variable2 <- accuracy.df$Accuracy[accuracy.df$City == name3 & accuracy.df$Algorithm == name4]
      variable <- c(variable1, variable2)
      #Analysis
      test.stat <- abs(mean(variable1) - mean(variable2)) #test-statistic
      B.samples <- matrix(sample(variable, size = n*B, replace=TRUE), #10,000 resamples with replacement
                            nrow=n, ncol=B)
      Boot.test.stat <- rep(0,B)
      for(i in 1:B){ #for each sample, find new test-statistic
        Boot.test.stat[i] <- abs(mean(B.samples[1:30,i]) - mean(B.samples[31:60,i])) #simulate difference in means to compare to test.stat
      }
      p.values[count] <- mean (Boot.test.stat >= test.stat) #record p-value, assuming H0: no difference in city-algo.
      
      cat(name1, "," , name2 , "," , name3 , "," , name4 , "i=" , count, "," , "obs. diff. in mean=" , test.stat, "\n")
      count <- count + 1
    }
  }
}
val <- abs(mean(accuracy.df$Accuracy[accuracy.df$City == "Mesa" & accuracy.df$Algorithm == "Random Forest"]) - 
             mean(accuracy.df$Accuracy[accuracy.df$City == "Palm Coast" & accuracy.df$Algorithm == "Random Forest"]))
hist(Boot.test.stat, main = "Bootstrap Distribution for RF-Mesa vs RF-Palm Coast", col = "light blue", xlab="Difference in Mean Accuracy", xlim=c(0,val+.001)) #final bootstrapped distribution
abline(v = val,col="red")
print(p.values)
p.adj <- p.adjust(p.values, method = "bonferroni", n = length(p.values))
print(p.adj) #adjust p-values to control for Family-wise error rate / Type I error
alpha <- 0.05
which(p.adj > alpha) #88 91
###88: Palm Coast-Neighbors vs. Austin-RF with adj. p-value = 0.105
###91: Austin-NN vs. Mesa-NN with adj. p-value = 0.5355
###are all NOT statistically significant, the rest are significant.

#Precision
n <- 30*2
B <- 10000
Cities <- c("Austin","Mesa","Palm_Coast")
Algorithms <- c("Bayes","Catboost","Neighbors","Neural_Network","Random_Forest")
veclist <- do.call(paste,expand.grid(Cities,Algorithms))
count <- 1
p.values <- rep(-1,105)
for(i in 1:15){
  temp <- strsplit(veclist[i], " ")[[1]]
  name1 <- gsub("_", " ", temp[1]) #City
  name2 <- gsub("_", " ", temp[2]) #Algo
  for(j in (i+1):15){
    temp2 <- strsplit(veclist[j], " ")[[1]]
    name3 <- gsub("_", " ", temp2[1]) #City
    name4 <- gsub("_", " ", temp2[2]) #Algo
    if( i != 15){ #prevent last two comparisons, so theres ((12-1)*12) / 2 = 66 total comparisons
      variable1 <- precision.df$Precision[precision.df$City == name1 & precision.df$Algorithm == name2]
      variable2 <- precision.df$Precision[precision.df$City == name3 & precision.df$Algorithm == name4]
      variable <- c(variable1, variable2)
      #Analysis
      test.stat <- abs(mean(variable1) - mean(variable2))
      B.samples <- matrix(sample(variable, size = n*B, replace=TRUE),
                          nrow=n, ncol=B)
      Boot.test.stat <- rep(0,B)
      for(i in 1:B){
        Boot.test.stat[i] <- abs(mean(B.samples[1:30,i]) - mean(B.samples[31:60,i]))
      }
      p.values[count] <- mean (Boot.test.stat >= test.stat)
      
      cat(name1, "," , name2 , "," , name3 , "," , name4 , "i=" , count, "," , "obs. diff. in mean=" , test.stat, "\n")
      count <- count + 1
    }
  }
}
print(p.values)
p.adj <- p.adjust(p.values, method = "bonferroni", n = length(p.values))
print(p.adj) #adjust p-values to control for Family-wise error rate / Type I error
alpha <- 0.05
which(p.adj > alpha) #78, 84, 91, 94, 98
###78: Mesa-Neighbors vs. Palm Coast-Neighbors with adj. p-value = 1.00
###84: Mesa-Neighbors vs. Palm Coast-RF with adj. p-value = 0.063
###91: Austin-NN vs. Mesa-NN with adj. p-value = 0.357
###94: Austin-NN vs. Mesa-RF with adj. p-value = 1.00
###98: Mesa-NN vs. Mesa-RF with adj. p-value = 0.1365
###are all NOT statistically significant, the rest are significant.

#Recall
n <- 30*2
B <- 10000
Cities <- c("Austin","Mesa","Palm_Coast")
Algorithms <- c("Bayes","Catboost","Neighbors","Neural_Network","Random_Forest")
veclist <- do.call(paste,expand.grid(Cities,Algorithms))
count <- 1
p.values <- rep(-1,105)
for(i in 1:15){
  temp <- strsplit(veclist[i], " ")[[1]]
  name1 <- gsub("_", " ", temp[1]) #City
  name2 <- gsub("_", " ", temp[2]) #Algo
  for(j in (i+1):15){
    temp2 <- strsplit(veclist[j], " ")[[1]]
    name3 <- gsub("_", " ", temp2[1]) #City
    name4 <- gsub("_", " ", temp2[2]) #Algo
    if( i != 15){ #prevent last two comparisons, so theres ((12-1)*12) / 2 = 66 total comparisons
      variable1 <- recall.df$Recall[recall.df$City == name1 & recall.df$Algorithm == name2]
      variable2 <- recall.df$Recall[recall.df$City == name3 & recall.df$Algorithm == name4]
      variable <- c(variable1, variable2)
      #Analysis
      test.stat <- abs(mean(variable1) - mean(variable2))
      B.samples <- matrix(sample(variable, size = n*B, replace=TRUE),
                          nrow=n, ncol=B)
      Boot.test.stat <- rep(0,B)
      for(i in 1:B){
        Boot.test.stat[i] <- abs(mean(B.samples[1:30,i]) - mean(B.samples[31:60,i]))
      }
      p.values[count] <- mean (Boot.test.stat >= test.stat)
      
      cat(name1, "," , name2 , "," , name3 , "," , name4 , "i=" , count, "," , "obs. diff. in mean=" , test.stat, "\n")
      count <- count + 1
    }
  }
}
print(p.values)
p.adj <- p.adjust(p.values, method = "bonferroni", n = length(p.values))
print(p.adj) #adjust p-values to control for Family-wise error rate / Type I error
alpha <- 0.05
which(p.adj > alpha) #48, 78, 96
###48: Austin-Catboost vs. Austin-RF with adj. p-value = 1.00
###78: Mesa-Neighbors vs. Palm Coast-Neighbors with adj. p-value = 1.00
###96: Mesa-NN vs. Palm Coast-NN with adj. p-value = 1.00
###are all NOT statistically significant, the rest are significant.

####FAILED METHODS BELOW###########################################
#LOG TRANSOFRMATIONS DIDN'T WORK

#new models
#data visualization
ggplot(accuracy.df, aes(log(Accuracy), Algorithm, colour = City)) + geom_point()
ggplot(precision.df, aes(log(Precision), Algorithm, colour = City)) + geom_point()
ggplot(recall.df, aes(log(Recall), Algorithm, colour = City)) + geom_point()
#models
accuracy.model <- aov(log(Accuracy) ~ City*Algorithm, data = accuracy.df)
precision.model <- aov(log(Precision) ~ City*Algorithm, data = precision.df)
recall.model <- aov(log(Recall) ~ City*Algorithm, data = recall.df)
#analysis of overall model
summary(accuracy.model)
summary(precision.model)
summary(recall.model)
#residual plots and assumptions
#1. Homogoneity of variance - Violated
#Because all n_i = 30, ANOVA ca be robust to small/medium violations
plot(accuracy.model,1)
plot(precision.model,1)
plot(recall.model,1)
leveneTest(log(Accuracy) ~ City*Algorithm, data = accuracy.df)
leveneTest(log(Precision) ~ City*Algorithm, data = precision.df)
leveneTest(log(Recall) ~ City*Algorithm, data = recall.df)
#2. Normality - Violated
plot(accuracy.model,2)
plot(precision.model,2)
plot(recall.model,2)
acc.res <- residuals(object = accuracy.model)
pre.res <- residuals(object = precision.model)
rec.res <- residuals(object = recall.model)
shapiro.test(x = acc.res)
shapiro.test(x = pre.res)
shapiro.test(x = rec.res)

#TRANSFORMATIONS DIDN'T WORK
#finding lambda
library(MASS)
out <- boxcox(lm(accuracy.df$Accuracy~accuracy.df$City*accuracy.df$Algorithm))
range(out$x[out$y > max(out$y)-qchisq(0.95,1)/2])
(-1.3535354 + 0.5454545) / 2

out.p <- boxcox(lm(precision.df$Precision~precision.df$City*precision.df$Algorithm))
range(out.p$x[out.p$y > max(out.p$y)-qchisq(0.95,1)/2])
(-2.000000 + -1.515152) / 2

out.r <- boxcox(lm(recall.df$Recall~recall.df$City*recall.df$Algorithm))
range(out.r$x[out.r$y > max(out.r$y)-qchisq(0.95,1)/2])
(-1.919192 + 1.070707) / 2
#new models
#data visualization
ggplot(accuracy.df, aes(Accuracy^(-1/2), Algorithm, colour = City)) + geom_point()
ggplot(precision.df, aes(Precision^(-1.75), Algorithm, colour = City)) + geom_point()
ggplot(recall.df, aes(Recall^(-1/2), Algorithm, colour = City)) + geom_point()
#models
accuracy.model <- aov(Accuracy^(-1/2) ~ City*Algorithm, data = accuracy.df)
precision.model <- aov(Precision^(-1.75) ~ City*Algorithm, data = precision.df)
recall.model <- aov(Recall^(-1/2) ~ City*Algorithm, data = recall.df)
#analysis of overall model
summary(accuracy.model)
summary(precision.model)
summary(recall.model)
#residual plots and assumptions
#1. Homogoneity of variance - Violated
#Because all n_i = 30, ANOVA ca be robust to small/medium violations
plot(accuracy.model,1)
plot(precision.model,1)
plot(recall.model,1)
leveneTest(Accuracy^(-1/2) ~ City*Algorithm, data = accuracy.df)
leveneTest(Precision^(-1.75) ~ City*Algorithm, data = precision.df)
leveneTest(Recall^(-1/2) ~ City*Algorithm, data = recall.df)
#2. Normality - Violated
plot(accuracy.model,2)
plot(precision.model,2)
plot(recall.model,2)
acc.res <- residuals(object = accuracy.model)
pre.res <- residuals(object = precision.model)
rec.res <- residuals(object = recall.model)
shapiro.test(x = acc.res)
shapiro.test(x = pre.res)
shapiro.test(x = rec.res)
