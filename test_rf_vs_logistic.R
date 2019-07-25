library(randomForest)
library(xgboost)
logistic <- function(x) exp(x) / (1 + exp(x))
logloss <- function (target, pred) { -mean(ifelse(target > 0, log(pred), log(1-pred)))}
acc <- function(target,pred){ sum(target==pred)/length(target)}
set.seed(1829)
n <- 10000
# x1 <- rnorm(n)
# x2 <- runif(n)
# x3 <- rpois(n, 4.5)
# inv_prob <- (-.2 + .3 * x1 + .1 * x2 + -.2 * x3 + # main effects
#                x1 * x2 + 1.4 * x2 * x3 + .1 * x1 * x2 * x3 - 3)/10# interactions 
x1 <- runif(n)
x2 <- runif(n)
inv_prob <- ( 1 * (x1 - x2))

y_prob <- logistic( inv_prob)
y <- rbinom(n, 1, y_prob)
train_frac <- .5
cases <- sample(seq_len(n), train_frac * n)
dat <- data.frame(x1, x2, y)
train <- dat[cases, ]
test <- dat[-cases, ]

x2_lr <- function(mdl, x1){
  m <- coefficients(mod_lr)
  -(m[['(Intercept)']] + x1*m[['x1']])/m[['x2']]
}
mod_lr <- glm(y ~ x1 + x2 , binomial, train)
predtr_lr <- predict(mod_lr, train, type = "response")
pred_lr <- predict(mod_lr, test, type = "response")
coefficients(mod_lr)
test$lr_prob <- pred_lr
test$lr_class <- test$lr_prob > .5

acc_lr <- acc(test$y, test$lr_class)

mod_rf <- randomForest(factor(y) ~ ., train)
pred_rf <- predict(mod_rf, test) == 1
test$rf <- pred_rf
acc_rf <- acc(test$y, test$rf)

acc_pop <- acc(test$y, (test$x1 - test$x2) > 0)
print( c(acc_lr, acc_rf, acc_pop))
dtrain <- xgb.DMatrix(as.matrix(train[c('x1', 'x2')]), label = train$y)
dtest <- xgb.DMatrix(as.matrix(test[c('x1', 'x2')]), label = test$y)
xgb_params = list(
  eta = 0.1,
  subsample=0.7,
  
  objective = "binary:logistic",
  max_depth=5,
  eval.metric = "logloss"
)

# cv <- xgb.cv(data = dtrain, nrounds = 50, nthread = 2, nfold = 5,
#              metrics = list("logloss"),
#              early_stopping_rounds=15,
#              max_depth = 5, eta = 0.05, objective = "binary:logistic")
bst <- xgb.train(xgb_params, data = dtrain, nrounds=100, 
                 watchlist=list(train=dtrain,test=dtest),
                 early_stopping_rounds=15)

predtr_bst <- predict(bst, as.matrix(train[c('x1', 'x2')]))
pred_bst <- predict(bst, as.matrix(test[c('x1', 'x2')]))



classes.train <- ifelse(train$x1>train$x2, "blue", "orange")
classes.test <- ifelse(test$x1>test$x2, "blue", "orange")
ran <- seq(0,1,0.01)
grid <- expand.grid(x1=ran, x2=ran)
classes.grid <- predict(mod_rf, grid) ==1
prob_lr.grid <- predict(mod_lr, grid, , type='response')
contour_levels <- seq(min(prob_lr.grid), max(prob_lr.grid), length=10)
prob_bst.grid <- predict(bst, as.matrix(grid))
# plot the boundary
# contour(x=ran, y=ran, z=matrix(classes.grid, nrow=length(ran)), levels=0.5,
#        col="grey", drawlabels=FALSE, lwd=2)
contour(x=ran, y=ran, z=matrix(classes.grid, nrow=length(ran)), levels=.5,
        col="grey", drawlabels=FALSE, lwd=2)

contour(x=ran, y=ran, z=matrix(prob_bst.grid, nrow=length(ran)), levels=contour_levels,
        col="grey", drawlabels=TRUE)
contour(x=ran, y=ran, z=matrix(prob_lr.grid, nrow=length(ran)), levels=contour_levels,
        col="grey", drawlabels=TRUE)

# add points from test dataset
points(test[c('x1','x2')], col=test$y*3 + 1)
#points(train[c('x1','x2')], col=train$y*3 + 2)
lines(c(0,1), c(0,1),col=10)
lines(c(0,1), x2_lr(mod_lr,c(0,1)), col=5)


plot(test$x1,test$x2, col=1 + test$lr_class)

title(paste('logistic regression ',paste(mod_lr$coefficients)))

plot(test$x1,test$x2, col=5 + test$y)
lines(c(0,1), c(0,1),col=1)
title('test data')

plot(test$x1,test$x2, col=3 + test$rf)
lines(c(0,1), c(0,1),col=1)
title('random forest')
cuts <- seq(0,1,.1) +0.05
midpoints <- seq(0,.9,.1) +0.05
test$x1c <- cut(test$x1,cuts)
test$x2c <- cut(test$x2,cuts)


