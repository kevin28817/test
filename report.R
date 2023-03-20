loans <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
library("skimr")
skim(loans)
summary(loans)
library(summarytools)
descr(loans)
library(ggplot2)
ggplot(loans,aes(x = 	Income, y = Personal.Loan)) +geom_point()
DataExplorer::plot_bar(loans, ncol = 3)
DataExplorer::plot_histogram(loans, ncol = 3)
DataExplorer::plot_boxplot(loans, by = "Personal.Loan", ncol = 3)

library("data.table")
library("mlr3verse")
library("tidyverse")
set.seed(777) # set seed for reproducibility
loans["Experience"][loans["Experience"] < 0] <- NA
loans <- na.omit(loans)
loans["Personal.Loan"] <- lapply(loans["Personal.Loan"], factor)
credit_task <- TaskClassif$new(id = "BankCredit",
                               backend = loans,
                               target = "Personal.Loan",
                               positive = "1")

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_lda  <- lrn("classif.lda", predict_type = "prob")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

set.seed(777)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(credit_task)
res <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_lda,
                    lrn_ranger,
                    lrn_xgboost,
                    lrn_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

set.seed(777)
rdesc <- rsmp("bootstrap", repeats = 2)
rdesc$instantiate(credit_task)
res1 <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_lda,
                    lrn_ranger,
                    lrn_xgboost,
                    lrn_log_reg),
  resampling = list(rdesc)
), store_models = TRUE)

res1$aggregate(list(msr("classif.ce"),
                    msr("classif.acc"),
                    msr("classif.fpr"),
                    msr("classif.fnr")))

set.seed(777)
repeated_cv <- rsmp("repeated_cv", folds = 4 , repeats = 2)
repeated_cv$instantiate(credit_task)
res2 <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_lda,
                    lrn_ranger,
                    lrn_xgboost,
                    lrn_log_reg),
  resampling = list(repeated_cv)
), store_models = TRUE)
res2$aggregate(list(msr("classif.ce"),
                    msr("classif.acc"),
                    msr("classif.fpr"),
                    msr("classif.fnr")))

cv <- list(res$aggregate(list(msr("classif.acc")))[[length(res$aggregate(list(msr("classif.acc"))))]])
bootstrap <- list(res1$aggregate(list(msr("classif.acc")))[[length(res1$aggregate(list(msr("classif.acc"))))]])
repeated_cv <- list(res2$aggregate(list(msr("classif.acc")))[[length(res2$aggregate(list(msr("classif.acc"))))]])
title <- c('classif.featureless','classif.rpart','classif.lda','classif.ranger','classif.xgboost','classif.log_reg')
df <- data.frame(cbind(cv[[1]], bootstrap[[1]],repeated_cv[[1]]))
colnames(df) <- c("cv", "bootstrap","repeated_cv")
rownames(df) <- title
df
cv1 <- list(res$aggregate(list(msr("classif.ce")))[[length(res$aggregate(list(msr("classif.ce"))))]])
bootstrap1 <- list(res1$aggregate(list(msr("classif.ce")))[[length(res1$aggregate(list(msr("classif.ce"))))]])
repeated_cv1 <- list(res2$aggregate(list(msr("classif.ce")))[[length(res2$aggregate(list(msr("classif.ce"))))]])
df1 <- data.frame(cbind(cv1[[1]], bootstrap1[[1]],repeated_cv1[[1]]))
colnames(df1) <- c("cv", "bootstrap","repeated_cv")
rownames(df1) <- title
df1

library(mlr3tuning)
library(mlr3)
set.seed(777)
learner = mlr_learners$get("classif.ranger")
ps = ParamSet$new(list(
  ParamInt$new("mtry", lower = 1L, upper = 4L),
  ParamInt$new("num.trees", lower = 10L, upper = 1000L)
))
instance <- TuningInstanceSingleCrit$new(
  task = credit_task,
  learner = learner,
  resampling = cv5,
  measure = msr("classif.acc"),
  terminator = trm("evals", n_evals = 25),
  search_space = ps
)
tuner <- tnr("grid_search", resolution = 5)
tuner$optimize(instance)

set.seed(777)
lrn_optimal <- lrn("classif.ranger", predict_type = "prob",mtry=4,num.trees=1000)
result <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_lda,
                    lrn_ranger,
                    lrn_xgboost,
                    lrn_log_reg,
                    lrn_optimal),
  resampling = list(cv5)
), store_models = TRUE)
result$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc"),
                      msr("classif.fpr"),
                      msr("classif.fnr")))

library(precrec)
set.seed(212)
lrn_optimal <- lrn("classif.ranger", predict_type = "prob",mtry=4,num.trees=257)
optimized_result <- resample(task = credit_task, learner = lrn_optimal, resampling = cv5, store_models = TRUE)
autoplot(optimized_result, measure = msr("classif.auc"))
autoplot(optimized_result,type = "roc",linewidth = 1)+ theme_bw()
autoplot(optimized_result, type = "prc")+ theme_bw()
train_data <- TaskClassif$new(id = "BankCredit",
                              backend = loans[optimized_result$resampling$test_set(1),],
                              target = "Personal.Loan",
                              positive = "1")
model <- lrn_optimal$train(train_data)
pred <- as.numeric(as.character(predict(model, loans)))
conf.mat <- table(`true loan` = loans$Personal.Loan, `predict loan` = pred )
conf.mat
cv_pred <- lrn_optimal$train(train_data)$predict(credit_task)
cv_pred_df <- as.data.table(cv_pred)
cali_df <- cv_pred_df %>% 
  arrange(prob.1) %>% 
  mutate(postive = if_else(truth == "1", 1, 0),
         group = c(rep(1:1237,each=4))
  ) %>% 
  group_by(group) %>% 
  summarise(mean_pred = mean(prob.1),
            mean_obs = mean(postive)
  )

cali_plot <- ggplot(cali_df, aes(mean_pred, mean_obs))+ 
  geom_point(alpha = 0.5)+
  geom_abline(linetype = "dashed")+
  theme_minimal()
cali_plot