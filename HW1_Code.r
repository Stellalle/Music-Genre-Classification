################################################################################################
## Load all the libraries
################################################################################################
library(R.matlab)
library(nnet)
library(caret)
library(ggplot2)
library(textir)
library(gbm)
library(glmnet)
library(deepnet)
library(randomForest)
library(e1071)
library(rpart)
library(Rmisc)
library(PRROC)
library(randomForest)

################################################################################################
## Load all the features
################################################################################################
FV.mat <- readMat('FV.mat')
LB.mat <- readMat('LB.mat')
FV.brightness.mat <- readMat('FV_brightness.mat')
FV.hcdf.mat <- readMat('FV_hcdf.mat')
FV.roughness.mat <- readMat('FV_roughness.mat')
FV.eng.mat <- readMat('FV_eng.mat')
FV.zerocross.mat <- readMat('FV_zerocross.mat')
genre <- factor(LB.mat$LB)

print(levels(genre))
print(length(genre))

df <- data.frame(genre, t(FV.mat$FV), t(FV.roughness.mat$FV.roughness))

################################################################################################
## Split data
################################################################################################
split.idx <- createDataPartition(df$genre, p = .9,
                                  list = FALSE,
                                  times = 1)
df.validation <- df[-split.idx,]
df <- df[split.idx,]
split.idx <- createDataPartition(df$genre, p=0.8, list=FALSE, times=1)
df.train <- df[split.idx,]
df.test <- df[-split.idx,]

print(dim(df.train))
print(dim(df.test))
print(dim(df.validation))

accuracy <- rep(0, 6)

################################################################################################
## Multinomial Logistic Regression
################################################################################################
nnet.fit <- multinom(genre ~. , data=df.train, MaxNWts=10000)

predicted.genre.softmax <- predict(nnet.fit, newdata=df.test)
predicted.genre.softmax.validation <- predict(nnet.fit, newdata=df.validation)

predicted.genre.softmax.validation.prob <- predict(nnet.fit, newdata=df.validation, class='response')

predicted.genre.softmax.validation.prob

print(length(which(predicted.genre.softmax == df.test$genre)) / length(df.test$genre))
print(length(which(predicted.genre.softmax.validation == df.validation$genre)) / length(df.validation$genre))

accuracy[1] <- length(which(predicted.genre.softmax.validation == df.validation$genre)) / length(df.validation$genre)

comp_plot <- function(pred, true_val){
  confusion <- as.data.frame(as.table(confusionMatrix(pred, true_val)))
  confusion[,3] <- (confusion[,3] - mean(confusion[,3])) / sqrt(var(confusion[,3]))
  plot <- ggplot(confusion)
  return(plot)
}

plot_softmax <- comp_plot(predicted.genre.softmax.validation, df.validation$genre)

################################################################################################
## Gradient Boosted Machine
################################################################################################
gbm.fit.model <- gbm(genre ~ ., data=df.train, cv.folds=5, n.trees = 500)
plot(gbm.fit.model)

predicted.genre.prob <- predict(gbm.fit.model, newdata=df.test, n.trees=500, type='response')


predicted.genre.prob.validation <- predict(gbm.fit.model, newdata=df.validation, n.trees=500, type='response')

predicted.genre.prob.validation <- predicted.genre.prob.validation[,,1]

predicted.genre.gbm <- apply(predicted.genre.prob, MARGIN=1, which.max)
predicted.genre.gbm.validation <- apply(predicted.genre.prob.validation, MARGIN=1, which.max)

print(length(which(predicted.genre.gbm == df.test$genre)) / length(df.test$genre))
print(length(which(predicted.genre.gbm.validation == df.validation$genre)) / length(df.validation$genre))

accuracy[2] <- length(which(predicted.genre.gbm.validation == df.validation$genre)) / length(df.validation$genre)
plot_gbm <- comp_plot(predicted.genre.gbm.validation, df.validation$genre)

################################################################################################
## Lasso Logistic Regression
################################################################################################
n = dim(df.train)[2]

glmnet.fitted.model <- glmnet(x=as.matrix(df.train[,2:n]),y=df.train[,1],
                              family="multinomial", lambda=0.04)

predicted.genre.lasso <- predict(glmnet.fitted.model, newx=as.matrix(df.test[,2:n]), type="class")
predicted.genre.lasso.validation <- predict(glmnet.fitted.model, newx=as.matrix(df.validation[,2:n]), type="class")

predicted.genre.lasso.validation.prob <- predict(glmnet.fitted.model, newx=as.matrix(df.validation[,2:n]), type="response")

predicted.genre.lasso.validation.prob <- predicted.genre.lasso.validation.prob[,,1]

print(length(which(predicted.genre.lasso == df.test$genre)) / length(df.test$genre))
print(length(which(predicted.genre.lasso.validation == df.validation$genre)) / length(df.validation$genre))

accuracy[3] <- length(which(predicted.genre.lasso.validation == df.validation$genre)) / length(df.validation$genre)
plot_lasso <- comp_plot(predicted.genre.lasso.validation, df.validation$genre)

################################################################################################
## Neural Networks
################################################################################################
y.target = matrix(as.matrix( rep(0, dim(df.train)[1]*10)), nrow=dim(df.train)[1], ncol= 10)
idx.x = seq_len(dim(y.target)[1])
idx.y <- as.numeric(df.train$genre)
for (i in seq_len(dim(y.target)[1])){
    y.target[i, idx.y[i]] <- 1
}

n = dim(df.train)[2]
x = unname(as.matrix(df.train[,2:n]))

dbn.fit.model <- nn.train(x, y.target, hidden = c(1000), activationfun = "tanh", learningrate = 0.5,
                               momentum = 0.5, learningrate_scale = 0.99, output = "softmax", numepochs = 80,
                               batchsize = 10, hidden_dropout = 0.6, visible_dropout = 0)

y.predicted <- nn.predict(dbn.fit.model, x=unname(as.matrix(df.test[,2:n])))
y.predicted.val <- nn.predict(dbn.fit.model, x=unname(as.matrix(df.validation[,2:n])))

predicted.genre.dnn <- apply(y.predicted, MARGIN=1, which.max)
predicted.genre.dnn.val <- apply(y.predicted.val, MARGIN=1, which.max)

y.predicted[1:3,]

print(length(which(predicted.genre.dnn == df.test$genre)) / length(df.test$genre))
print(length(which(predicted.genre.dnn.val == df.validation$genre)) / length(df.validation$genre))

accuracy[4] <- length(which(predicted.genre.dnn.val == df.validation$genre)) / length(df.validation$genre)
plot_dnn <- comp_plot(predicted.genre.dnn.val, df.validation$genre)

################################################################################################
## Support Vector Machines
################################################################################################
svm.model <- svm(genre ~ ., data=df.train, cv.folds=5)
predicted.genre.svm.validation <- predict(svm.model, df.validation)
summary(svm.model)

track_error <- function(pred, true_val){
  track <- rep(0, 10)
  for (i in 1:length(true_val)){
    if (pred[i] != true_val[i]){
      index <- as.integer(true_val[i])
      track[index] <- track[index] + 1
    }
  }
  return(track)
}

print(length(which(predicted.genre.svm.validation == df.validation$genre)) / length(df.validation$genre))

accuracy[5] <- length(which(predicted.genre.svm.validation == df.validation$genre)) / length(df.validation$genre)
plot_svm <- comp_plot(predicted.genre.svm.validation, df.validation$genre)

################################################################################################
## Random Forest
################################################################################################
rf.model <- randomForest(x = df.train[,2:n], y = df.train[,1], importance = TRUE)
varImpPlot(rf.model)
predicted.genre.rf.validation <- predict(rf.model, newdata = as.matrix(df.validation[,2:n]), type="response", predict.all = FALSE)

print(length(which(predicted.genre.rf.validation == df.validation$genre)) / length(df.validation$genre))

genre.val <- as.matrix(df.validation$genre)
accuracy[6] <- length(which(predicted.genre.rf.validation == genre.val)) / length(df.validation$genre)
plot_rf <- comp_plot(predicted.genre.rf.validation, df.validation$genre)


Ntrees <- c(1, seq(2,8,2), seq(10, 100, 10))
rf.results <- rep(0, length(Ntrees))
for (i in 1:length(Ntrees)){
  rf.model <- randomForest(x = df.train[,2:n], y = df.train[,1], ntree = Ntrees[i])
  predicted.genre.rf.validation <- predict(rf.model, newdata = as.matrix(df.validation[,2:n]), type="response", predict.all = FALSE)
  rf.results[i] = length(which(predicted.genre.rf.validation == df.validation$genre)) / length(df.validation$genre)
}

length(Ntrees)
length(rf.results)
rf.model.plot <- data.frame(Ntrees, rf.results)
ggplot(rf.model.plot, aes(x = Ntrees, y = rf.results)) + geom_line(stat = "identity") + geom_text(aes(y=rf.results, ymax=rf.results, label=round(rf.results, 2)), vjust=-.5, color="black") + ggtitle('Prediction Accuracy with Different Number of Trees in Random Forest') + scale_y_continuous("Prediction Accuracy",limits=c(0.3, 0.7),breaks=seq(0.3, 0.7, .1))

################################################################################################
## Plots
################################################################################################
classifier <- c("MLR", "GBM", "Lasso", "NN", "SVM", "RF")
df.accuracy <- data.frame(classifier, accuracy)

p0 <- ggplot(df.accuracy, aes(x = classifier, y = accuracy)) + geom_bar(stat = "identity", position=position_dodge()) + geom_text(aes(y=accuracy, ymax=accuracy, label=round(accuracy, 4)), position= position_dodge(width=0.9), vjust=-.5, color="black") + ggtitle('Prediction Accuracy') + scale_y_continuous("Prediction Accuracy",limits=c(0, 0.8),breaks=seq(0, 0.8, .2))

p1 <- plot_softmax + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Multinomial Logistic Regression')
p2 <- plot_gbm + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Gradient Boosted Machine')
p3 <- plot_lasso + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Lasso Logistic Regression')
p4 <- plot_dnn + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Neural Networks')
p5 <- plot_svm + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Support Vector Machine')
p6 <- plot_rf + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Genre") + scale_y_discrete(name="Predicted Genre") + scale_fill_gradient(breaks=seq(from=-1, to=5, by=0.5)) + ggtitle('Random Forest')

multiplot(p1, p2, p3, p4, p5, p6, cols=2)

################################################################################################
## Emsemble Classifier
################################################################################################
df.ensemble <- data.frame(genre=as.factor(df.test$genre), softmax = as.factor(predicted.genre.softmax), 
                          gbm = as.factor(predicted.genre.gbm), 
                          lasso = as.factor(predicted.genre.lasso), 
                          dnn=as.factor(predicted.genre.dnn))

df.ensemble.validataion <- data.frame(genre=as.factor(df.validation$genre), 
                                      softmax = as.factor(predicted.genre.softmax.validation), 
                          gbm = as.factor(predicted.genre.gbm.validation), 
                          lasso = as.factor(predicted.genre.lasso.validation), 
                          dnn=as.factor(predicted.genre.dnn.val))



ensemble.fit <- gbm(genre ~  lasso+ dnn, 
                    data=rbind( df.ensemble,df.ensemble.validataion),
                    interaction.depth = 1,
                    distribution="multinomial")

ensemble.fit.lm <- lm(genre ~  lasso  + dnn, data=rbind(df.ensemble,df.ensemble.validataion) )

summary(ensemble.fit)

ensemble.predicted.prob <- predict(ensemble.fit, newx = df.ensemble.validataion, n.trees=80, type='response')

ensemble.predicted.prob <- ensemble.predicted.prob[,,1]

dim(ensemble.predicted.prob)

ensemble.predicted.prob[1:3,]

ensemble.predicted.class <- apply(ensemble.predicted.prob, MARGIN=1, which.max)

print(length(which(ensemble.predicted.class == df.ensemble.validataion$genre)) / length(df.ensemble.validataion$genre))

ensemble.predicted.class <- rep(0, dim(df.validation)[1])

predicted.genre.dnn.val

for (i in seq_len(dim(df.validation)[1])){
    vote = rep(0, 10)
    vote[predicted.genre.softmax.validation[i]] = vote[predicted.genre.softmax.validation[i]] + 1
    vote[as.numeric(predicted.genre.lasso.validation[i])] = vote[as.numeric(predicted.genre.lasso.validation[i])] + 1
    #vote[predicted.genre.gbm.validation[i]] = vote[predicted.genre.gbm.validation[i]] + 1
    vote[predicted.genre.dnn.val[i]] = vote[predicted.genre.dnn.val[i]] + 1
    
    print(vote)
    if (max(vote)==1){
        ensemble.predicted.class[i] <- predicted.genre.dnn.val[i]
    } else {
        ensemble.predicted.class[i] <- which.max(vote)
    }
}

print(length(which(ensemble.predicted.class == df.ensemble.validataion$genre)) / length(df.ensemble.validataion$genre))

ensemble.predicted.prob <- predicted.genre.lasso.validation.prob + predicted.genre.prob.validation + y.predicted.val

ensemble.predicted.prob[1:3,]

ensemble.predicted.class <- apply(ensemble.predicted.prob, MARGIN=1, which.max)

print(length(which(ensemble.predicted.class == df.ensemble.validataion$genre)) / length(df.ensemble.validataion$genre))


