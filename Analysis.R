################################################################################
# R script to accompany Phillips et al.                                      #
# Generates GLMs in the frequentist and Bayesian paradigms to estimate         #
# seafood mislabelling rates in metro Vancouver, British Columbia, Canada      #
# based on Hu et al. (2018)                                                    #
################################################################################

##### Set working directory #####

setwd("/Users/jarrettphillips/desktop/Seafood Fraud Modelling Paper")


##### Install required packages #####

# install.packages("rstan") 
#install.packages("rstanarm")
# install.packages("bayesplot") 
# install.packages("car") 
# install.packages("dplyr")

##### Load required libraries #####

library(rstan) # for MCMC
library(rstanarm) # for Bayesian GLMs
library(bayesplot) # For posterior visualization
library(car) # for VIF
library(dplyr) # for data summary


options(mc.cores = parallel::detectCores()) # parallelize simulations
rstan_options(auto_write = TRUE) # only have to compile once unless code is changed


##### Import data #####

x <- read.csv(file.choose())
x <- na.omit(x) # remove rows with NA


##### Preprocessing data #####

# Trim whitespace

x$Source <- trimws(x$Source)
x$State <- trimws(x$State)
x$Appearance <- trimws(x$Appearance)
x$Form <- trimws(x$Form)
x$Colour <- trimws(x$Colour)

# Convert categorical variables to factors

Source <- as.factor(x$Source)
State <- as.factor(x$State)
Appearance <- as.factor(x$Appearance)
Form <- as.factor(x$Form)
Colour <- as.factor(x$Colour)

Mislabelled <- x$Mislabelled # response variable

df <- cbind.data.frame(Source = Source,
            State = State,
            Appearance = Appearance,
            Form = Form,
            Colour = Colour,
            Mislabelled = Mislabelled)


##### Data Summary #####

df |> 
  count(Source, Mislabelled)

df |> 
  count(State, Mislabelled)

df |> 
  count(Appearance, Mislabelled)

df |> 
  count(Form, Mislabelled)

df |> 
  count(Colour, Mislabelled)


##### Frequentist Logistic GLM #####

mod <- glm(Mislabelled ~ . , data = df, family = binomial(link = "logit")) # logistic GLM
summary(mod) # model summary

(vifs <- round(vif(mod), 3)) # predictor variance inflation factors

(ci_log_odds <- round(confint(mod), 3)) # log odds CI

(odds <- round(exp(mod$coefficients), 3)) # odds
(ci_odds <- round(exp(ci_log_odds), 3)) # odds CI

(prob <- round((odds / (1 + odds)), 3)) # probabilities
(ci_prob <- round((ci_odds / (1 + ci_odds)), 3))  # probabilities CI


##### Bayesian Logistic GLM in rstanarm #####

# initialize chains to MLEs

inits <- function(...) {
  list("(Intercept)" =  2.23492010,
              "SourceRestaurant" =  0.89308118,
              "SourceSushiBar" = 0.06102637,
              "StateRaw" = -1.18759571,
              "AppearancePlain" = 1.84558791,
              "FormChunk" = -3.57801992,
              "FormFillet" = -3.55850830,
              "FormWhole" = -1.57447378, 
              "ColourRed" = -1.74047158)
}

seed <- 0673227 # set seed for reproducibility

# flat (uniform) prior on intercept, with N(0, 2.5) on coefficients
fit1 <- stan_glm(Mislabelled ~ ., 
                 family = binomial(), 
                 data = df, 
                 prior = NULL,
                 init = inits,
                 seed = seed)

draws_fit1 <- as.data.frame(fit1) # posterior samples
(odds_fit1 <- round(exp(colMeans(draws_fit1)), 3))
(prob_fit1 <- round((exp(colMeans(draws_fit1)) / (1 + exp(colMeans(draws_fit1)))), 3))


# N(0, 1) prior on intercepts and coefficients
fit2 <- stan_glm(Mislabelled ~ ., 
                 family = binomial(), 
                 data = df, 
                 prior = normal(0, 1), 
                 prior_intercept = normal(0, 1),
                 init = inits,
                 seed = seed)

draws_fit2 <- as.data.frame(fit2)
(odds_fit2 <- round(exp(colMeans(draws_fit2)), 3))
(prob_fit2 <- round((exp(colMeans(draws_fit2)) / (1 + exp(colMeans(draws_fit2)))), 3))



# default prior - N(0, 2.5) on intercept and coefficients
fit3 <- stan_glm(Mislabelled ~ ., 
                 family = binomial(), 
                 data = df,
                 init = inits,
                 seed = seed)

draws_fit3 <- as.data.frame(fit3)
(odds_fit3 <- round(exp(colMeans(draws_fit3)), 3))
(prob_fit3 <- round((exp(colMeans(draws_fit3)) / (1 + exp(colMeans(draws_fit3)))), 3))

# Cauchy prior -  Cau(0, 10) on intercept and Cau(0, 2.5) on coefficients
fit4 <- stan_glm(Mislabelled ~ ., 
                 family = binomial(), 
                 data = df, 
                 prior = cauchy(0, 2.5), 
                 prior_intercept = cauchy(0, 10),
                 init = inits,
                 seed = seed)

draws_fit4 <- as.data.frame(fit4)
(odds_fit4 <- round(exp(colMeans(draws_fit4)), 3))
(prob_fit4 <- round((exp(colMeans(draws_fit4)) / (1 + exp(colMeans(draws_fit4)))), 3))


# Get a summary of fit1
summary_fit1 <- fit1$stan_summary[, c("mean", "sd", "2.5%", "97.5%")]
(summary_fit1 <- round(summary_fit1, 3))

plot(fit1, "trace")

(odds_cri_fit1 <- round(exp(posterior_interval(fit1, prob = 0.95)), 3)) # odds credible interval
(probs_cri_fit1 <- round(exp(posterior_interval(fit1, prob = 0.95)) / (1 + exp(posterior_interval(fit1, prob = 0.95))), 3)) # probability credible interval


# fit2
summary_fit2 <- fit2$stan_summary[, c("mean", "sd", "2.5%", "97.5%")]
(summary_fit2 <- round(summary_fit2, 3))

plot(fit2, "trace")

(odds_cri_fit2 <- round(exp(posterior_interval(fit2, prob = 0.95)), 3)) # odds credible interval
(probs_cri_fit2 <- round(exp(posterior_interval(fit2, prob = 0.95)) / (1 + exp(posterior_interval(fit2, prob = 0.95))), 3)) # probability credible interval


# fit3
summary_fit3 <- fit3$stan_summary[, c("mean", "sd", "2.5%", "97.5%")]
(summary_fit3 <- round(summary_fit3, 3))

plot(fit3, "trace")

(odds_cri_fit3 <- round(exp(posterior_interval(fit3, prob = 0.95)), 3)) # odds credible interval
(probs_cri_fit3 <- round(exp(posterior_interval(fit3, prob = 0.95)) / (1 + exp(posterior_interval(fit4, prob = 0.95))), 3)) # probability credible interval


# fit4
summary_fit4 <- fit4$stan_summary[, c("mean", "sd", "2.5%", "97.5%")]
(summary_fit4 <- round(summary_fit4, 3))

plot(fit4, "trace")

(odds_cri_fit4 <- round(exp(posterior_interval(fit4, prob = 0.95)), 3)) # odds credible interval
(probs_cri_fit4 <- round(exp(posterior_interval(fit4, prob = 0.95)) / (1 + exp(posterior_interval(fit4, prob = 0.95))), 3)) # probability credible interval


### Posterior Predictive Checks ###

pp_check(fit1, seed = seed)
pp_check(fit2, seed = seed)
pp_check(fit3, seed = seed)
pp_check(fit4, seed = seed)

ppc_stat(Mislabelled, posterior_predict(fit1), stat = "mean")
ppc_stat(Mislabelled, posterior_predict(fit2), stat = "mean")
ppc_stat(Mislabelled, posterior_predict(fit3), stat = "mean")
ppc_stat(Mislabelled, posterior_predict(fit4), stat = "mean")


### Prediction and Classifixcation ###

G80 <- data.frame(
  Source = "Grocery",
  State = "Raw",
  Appearance = "Modified",
  Form = "Fillet",
  Colour = "Light"
)

R2 <- data.frame(
  Source = "Restaurant",
  State = "Cooked",
  Appearance = "Modified",
  Form = "Fillet",
  Colour = "Red"
)

R51 <- data.frame(
  Source = "Restaurant",
  State = "Cooked",
  Appearance = "Modified",
  Form = "Fillet",
  Colour = "Light"
)

S10 <- data.frame(
  Source = "SushiBar",
  State = "Raw",
  Appearance = "Plain",
  Form = "Fillet",
  Colour = "Light"
)

## frequentist ##

(pred_G80_freq <- round(predict(mod, newdata = G80, type = "response"), 3))
(pred_R2_freq <- round(predict(mod, newdata = R2, type = "response"), 3)) 
(pred_R51_freq <- round(predict(mod, newdata = R51, type = "response"), 3))
(pred_S10_freq <- round(predict(mod, newdata = S10, type = "response"), 3))

(log_odds_G80_freq <- round(predict(mod, newdata = G80, type = "link"), 3))
(log_odds_R2_freq <- round(predict(mod, newdata = R2, type = "link"), 3))
(log_odds_R51_freq <- round(predict(mod, newdata = R51, type = "link"), 3))
(log_odds_S10_freq <- round(predict(mod, newdata = S10, type = "link"), 3))

(odds_G80_freq <- round(exp(predict(mod, newdata = G80, type = "link")), 3))
(odds_R2_freq <- round(exp(predict(mod, newdata = R2, type = "link")), 3)) 
(odds_R51_freq <- round(exp(predict(mod, newdata = R51, type = "link")), 3))
(odds_S10_freq <- round(exp(predict(mod, newdata = S10, type = "link")), 3))


# Same as above #

# (odds_G80_freq <- round(predict(mod, newdata = G80, type = "response") / (1 - predict(mod, newdata = G80, type = "response")), 3))
# (odds_R2_freq <- round(predict(mod, newdata = R2, type = "response") / (1 - predict(mod, newdata = R2, type = "response")), 3))
# (odds_R51_freq <- round(predict(mod, newdata = R51, type = "response") / (1 - predict(mod, newdata = R51, type = "response")), 3))
# (odds_S10_freq <- round(predict(mod, newdata = S10, type = "response") / (1 - predict(mod, newdata = S10, type = "response")), 3))
# 
# (log_odds_G80_freq <- round(log(predict(mod, newdata = G80, type = "response") / (1 - predict(mod, newdata = G80, type = "response"))), 3))
# (log_odds_R2_freq <- round(log(predict(mod, newdata = R2, type = "response") / (1 - predict(mod, newdata = R2, type = "response"))), 3))
# (log_odds_R51_freq <- round(log(predict(mod, newdata = R51, type = "response") / (1 - predict(mod, newdata = R51, type = "response"))), 3))
# (log_odds_S10_freq <- round(log(predict(mod, newdata = S10, type = "response") / (1 - predict(mod, newdata = S10, type = "response"))), 3))


## Bayesian ##

(pred_G80_bayes_fit1 <- round(mean(posterior_predict(fit1, newdata = G80, seed = seed)), 3))
(pred_R2_bayes_fit1 <- round(mean(posterior_predict(fit1, newdata = R2, seed = seed)), 3))
(pred_R51_bayes_fit1 <- round(mean(posterior_predict(fit1, newdata = R51, seed = seed)), 3))
(pred_S10_bayes_fit1 <- round(mean(posterior_predict(fit1, newdata = S10, seed = seed)), 3))

(pred_G80_bayes_fit2 <- round(mean(posterior_predict(fit2, newdata = G80, seed = seed)), 3))
(pred_R2_bayes_fit2 <- round(mean(posterior_predict(fit2, newdata = R2, seed = seed)), 3))
(pred_R51_bayes_fit2 <- round(mean(posterior_predict(fit2, newdata = R51, seed = seed)), 3))
(pred_S10_bayes_fit2 <- round(mean(posterior_predict(fit2, newdata = S10, seed = seed)), 3))

(pred_G80_bayes_fit3 <- round(mean(posterior_predict(fit3, newdata = G80, seed = seed)), 3))
(pred_R2_bayes_fit3 <- round(mean(posterior_predict(fit3, newdata = R2, seed = seed)), 3))
(pred_R51_bayes_fit3 <- round(mean(posterior_predict(fit3, newdata = R51, seed = seed)), 3))
(pred_S10_bayes_fit3 <- round(mean(posterior_predict(fit3, newdata = S10, seed = seed)), 3))

(pred_G80_bayes_fit4 <- round(mean(posterior_predict(fit4, newdata = G80, seed = seed)), 3))
(pred_R2_bayes_fit4 <- round(mean(posterior_predict(fit4, newdata = R2, seed = seed)), 3))
(pred_R51_bayes_fit4 <- round(mean(posterior_predict(fit4, newdata = R51, seed = seed)), 3))
(pred_S10_bayes_fit4 <- round(mean(posterior_predict(fit4, newdata = S10, seed = seed)), 3))

(log_odds_G80_bayes_fit1 <- round(logit(pred_G80_bayes_fit1), 3))
(log_odds_R2_bayes_fit1 <- round(logit(pred_R2_bayes_fit1), 3))
(log_odds_R51_bayes_fit1 <- round(logit(pred_R51_bayes_fit1), 3))
(log_odds_S10_bayes_fit1 <- round(logit(pred_S10_bayes_fit1), 3))

(log_odds_G80_bayes_fit2 <- round(logit(pred_G80_bayes_fit2), 3))
(log_odds_R2_bayes_fit2 <- round(logit(pred_R2_bayes_fit2), 3))
(log_odds_R51_bayes_fit2 <- round(logit(pred_R51_bayes_fit2), 3))
(log_odds_S10_bayes_fit2 <- round(logit(pred_S10_bayes_fit2), 3))

(log_odds_G80_bayes_fit3 <- round(logit(pred_G80_bayes_fit3), 3))
(log_odds_R2_bayes_fit3 <- round(logit(pred_R2_bayes_fit3), 3))
(log_odds_R51_bayes_fit3 <- round(logit(pred_R51_bayes_fit3), 3))
(log_odds_S10_bayes_fit3 <- round(logit(pred_S10_bayes_fit3), 3))

(log_odds_G80_bayes_fit4 <- round(logit(pred_G80_bayes_fit4), 3))
(log_odds_R2_bayes_fit4 <- round(logit(pred_R2_bayes_fit4), 3))
(log_odds_R51_bayes_fit4 <- round(logit(pred_R51_bayes_fit4), 3))
(log_odds_S10_bayes_fit4 <- round(logit(pred_S10_bayes_fit4), 3))

(odds_G80_bayes_fit1 <- round(exp(log_odds_G80_bayes_fit1), 3))
(odds_R2_bayes_fit1 <- round(exp(log_odds_R2_bayes_fit1), 3))
(odds_R51_bayes_fit1 <- round(exp(log_odds_R51_bayes_fit1), 3))
(odds_S10_bayes_fit1 <- round(exp(log_odds_S10_bayes_fit1), 3))

(odds_G80_bayes_fit2 <- round(exp(log_odds_G80_bayes_fit2), 3))
(odds_R2_bayes_fit2 <- round(exp(log_odds_R2_bayes_fit2), 3))
(odds_R51_bayes_fit2 <- round(exp(log_odds_R51_bayes_fit2), 3))
(odds_S10_bayes_fit2 <- round(exp(log_odds_S10_bayes_fit2), 3))

(odds_G80_bayes_fit3 <- round(exp(log_odds_G80_bayes_fit3), 3))
(odds_R2_bayes_fit3 <- round(exp(log_odds_R2_bayes_fit3), 3))
(odds_R51_bayes_fit3 <- round(exp(log_odds_R51_bayes_fit3), 3))
(odds_S10_bayes_fit3 <- round(exp(log_odds_S10_bayes_fit3), 3))

(odds_G80_bayes_fit4 <- round(exp(log_odds_G80_bayes_fit4), 3))
(odds_R2_bayes_fit4 <- round(exp(log_odds_R2_bayes_fit4), 3))
(odds_R51_bayes_fit4 <- round(exp(log_odds_R51_bayes_fit4), 3))
(odds_S10_bayes_fit4 <- round(exp(log_odds_S10_bayes_fit4), 3))
