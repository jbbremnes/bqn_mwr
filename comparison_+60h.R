##    R code applied in Bremnes(2020) to make quantile forecasts for QFRNN(BQN), CQRS and QRNN
##    on the test data (year 2015). Note that data is not available.
##
##    Bremnes, J. B. (2020). Ensemble Postprocessing Using Quantile Function Regression Based on
##        Neural Networks and Bernstein Polynomials, Monthly Weather Review, 148(1), 403-414
##        https://journals.ametsoc.org/view/journals/mwre/148/1/mwr-d-19-0227.1.xml
##
##    June 2019  John Bjørnar Bremnes
##


##
##     O R G A N I Z E     D A T A 
##

##  read data
load("/disk1/prosjekt/qr_splines_wind/ens+obs_00+060.RData")

##  create dataframe with ordered members
kens       <- paste("FF", 0:50, sep=".")
xs         <- x[, c("SITE", "TIME", "FX_1", kens, "MEAN", "SD")]
xs$sid     <- match(xs$SITE, unique(xs$SITE)) - 1    ## in keras 0 is the first!
xs[, kens] <- scale( t(apply(xs[, kens], 1, sort)) )

##  training and test sets
kt   <- x$TIME < 2015010100  ## 49883 cases
kp   <- !kt                  ## 45477 cases



##
##     D E F I N I T I O N S
##

##  quantile levels for prediction
prob_out <- 1:51 / 52

##  predicted quantiles
qts <- list(RAW = t( apply(x[kp, kens], 1, sort) ))




##
##      N E U R A L     N E T     P R E D I C T I O N S
##

date()

##  number of epochs
epochs <- 90

##  quantile levels for loss function
prob   <- prob_out

##  degree of Bernstein polynomials
degree <- 8

##  function for Bernstein polynomials
Bern <- function(n, prob)
    return( sapply(0:n, dbinom, size = n, prob = prob) )

##  quantile loss function
qt_loss <- function(obs, b, degree, prob) {
    np   <- length(prob)
    B    <- Bern(degree, prob)     ## for Keras, likely better to compute B only once
    qts  <- k_dot(b, k_constant(as.numeric(B), shape = c(degree+1,np)))  ##  quantiles
    err  <- obs - qts
    e1   <- err * k_constant(prob, shape = c(1, np))
    e2   <- err * k_constant(prob - 1, shape = c(1, np))
    k_mean( k_maximum(e1, e2), axis = 2 ) 
}


##  lists to store fitted Bernstein coefficients  
b <- list() 

##  fit 10 neural networks
library(keras)
set.seed(pi)  ## no effect?
qts$NN <- 0

date()
for (i in paste("NN", 1:10, sep = "_")) {

    cat(i, "... ")
    input_sid <- layer_input(shape = 1)
    input_ens <- layer_input(shape = length(kens))
    embed_sid <- input_sid %>%
        layer_embedding(input_dim = length(unique(xs$sid)), output_dim = 8, input_length = 1) %>%
        layer_flatten()
    output    <- layer_concatenate(c(input_ens, embed_sid)) %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 32, activation = "relu") %>%
        layer_dense(units = degree + 1)
    model  <- keras_model(inputs = c(input_ens, input_sid), outputs = output)
    model %>%
        compile(optimizer = "adam", loss = function(y_true, y_pred)
            qt_loss(y_true, y_pred, degree, prob))
    
    nnfit   <- model %>% fit(x = list(as.matrix(xs[kt,kens]), xs$sid[kt]), y = xs$FX_1[kt],
                             epochs = epochs, batch_size = 128, verbose = 0)

    ##  make predictions of the Bernstein coefficients
    b[[i]]   <- model %>% predict(x = list(as.matrix(xs[kp,kens]), xs$sid[kp]))
    
    ##  compute quantiles
    qts[[i]]   <- b[[i]] %*% t(Bern(n = degree, prob = prob_out))
    qts$NN     <- qts$NN + qts[[i]] 
    
    cat("ok\n")
    rm(model, nnfit)
    gc()
    
} ##  7 minutes on 12 cores (but only at ~15% load) (with GPU + 15% CPU => ~13 minutes) 
date()

##  average neural net predictions
qts$NN <- qts$NN / 10




##
##     C Q R S     P R E D I C T I O N S 
##

source("~/prosjekt/qr_splines_wind/cqrs_ensemble.R")
library(quantreg)
library(splines)

qts$CQRS <- matrix(NA, nrow = sum(kp), ncol = length(kens))

xp       <- (x[ , kens] + x[, "MEAN"]) / 2
date()
for (s in unique(x$SITE)) {

    ks   <- x$SITE == s
    kt0  <- kt & ks
    kp0  <- kp & ks
    kqts <- x$SITE[kp] == s
    
    ##  define upper limit
    upper <- max(x$FX_1[kt0], na.rm=TRUE) * 1.3

    ##  constrained quantile regression splines with 1 knot and extrapolation
    fitCQRS          <- cqrs_ensemble.fit(ens=xp[kt0, kens], obs=x[kt0,"FX_1"], knots.interior=1,
                                          degree=3, reorder=TRUE, c0=c(0,upper), c1=c(0,NA),
                                          increasing=TRUE)
    qts$CQRS[kqts, ] <- cqrs_ensemble.predict(fit=fitCQRS, ens=xp[kp0, kens],
                                              reorder=TRUE, extrapolation=TRUE)
}
date()  ## 13 secs on single core





##
##     Q U A N T I L E     R E G R E S S I O N     N E U R A L     N E T W O R K
##

library(qrnn)

prob_qrnn <- seq(1, 51, by = 5) / 52    ## more is very slow!

qts$QRNN  <- matrix(NA, nrow = sum(kp), ncol = length(kens))

date()
for (s in unique(x$SITE)) {
    cat(s, " ")
    ks   <- x$SITE == s
    kt0  <- kt & ks
    kp0  <- kp & ks
    kqts <- x$SITE[kp] == s

    fitQRNN          <- mcqrnn.fit(x = as.matrix(x[kt0, c("MEAN","SD")]),
                                   y = as.matrix(x[kt0, "FX_1"]),
                                   n.hidden = 2,
                                   tau = prob_qrnn, monotone = 1, trace = FALSE)
    qts$QRNN[kqts, ] <- mcqrnn.predict(x = as.matrix(x[kp0, c("MEAN","SD")]),
                                       parms = fitQRNN, tau = prob_out)    
} ##  2040 secs on 1 core
cat("\n")
date()



##
##     S A V E     D A T A 
##

info <- x[kp, ]
obs  <- x[kp, "FX_1"]

save(prob_out, b, qts, obs, info, file="data/exp_comparison_+60h.RData")



##
##     B A S I C     V A L I D A T I O N
##

qs <- sapply(qts, function(u)
    colMeans( ((u > obs) - rep(prob_out, each=nrow(u))) * (u - obs) ))
qss <- (1 - qs / qs[, rep("CQRS", length(qts))] ) * 100 
matplot(prob_out, qss[, -1], type = "l", ylab = "QSS (%)", las = 1, ylim=c(-10,5))

(1 - colMeans(qs)/mean(qs[,"CQRS"])) * 100  





