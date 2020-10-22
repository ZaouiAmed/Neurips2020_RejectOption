########################################################################
########################################################################
##Illustration of the proposed procedure with KNN algorithm 
##Application to Boston house price dataset 
##We consider the desired rejection rate fixed to 0.3 and
##We fix the parameter k of KNN function equal to 10
##In order to evaluate error rate and rejection rate, we consider 
##30 repetitions of the scheme described in Section 5
########################################################################
########################################################################

rm(list=ls())
set.seed(1234)
## required packages##
library(FNN) ##for KNN function
library(MASS) ##for Boston dataset


#####################################################################################################
##Function decompData used to split the data into three subsets (Train, Unlabeled and Validation)
#####################################################################################################
decompData <- function(sampleDat, sizeT = 256, sizeU = 100){

sizeV <- nrow(sampleDat) - sizeT - sizeU

idx <- c(rep(1, sizeT), rep(2,sizeU),rep(3,sizeV))
idx.pert <- sample(idx, replace = F)
datT <- list(X = sampleDat[which(idx.pert == 1), 1:13], Y = sampleDat[which(idx.pert == 1),14])
datU <- list(X = sampleDat[which(idx.pert == 2), 1:13], Y = sampleDat[which(idx.pert == 2),14])
datV <- list(X = sampleDat[which(idx.pert == 3), 1:13], Y = sampleDat[which(idx.pert == 3),14])

return(list(datT = datT, datU = datU, datV = datV)) 
}

######################################################################################################
######################################################################################################



data(Boston) ##load the dataset
sampleData <- Boston
colnames(sampleData) <- c(sapply(1:13, function(int){paste("X", int, sep = "")}), "Y") ##X the features, Y the output
sampleData <- na.omit(sampleData)
 
eps <- 0.3 ##desired rejection rate
paramKnn <- 10 ##parameter k for KNN algorithm

results <- matrix(NA, ncol = 2, nrow = 30)
colnames(results) <- c("error rate", "rejection rate")
 

for (Nrep in 1:30){
     
     dat <- decompData(sampleData)
     dataTrain     <- dat$datT
     dataUnlabeled <- dat$datU$X
     dataTest      <- dat$datV
    
     pred_Knn <- knn.reg(train = as.matrix(dataTrain$X), test = as.matrix(dataTrain$X), k = paramKnn , y = dataTrain$Y)$pred ##computation of \hat{f}(X_i)
     YRes <- (dataTrain$Y - pred_Knn)^2 ## computation of the residuals (Y_i-\hat{f}(X_i))^2

##Based on the unlabeled dataset, computation of the empirical ecdf with small perturbation
     sigmaKnnPredUnlabeled <- knn.reg(train  = as.matrix(dataTrain$X), test  = as.matrix(dataUnlabeled), k = paramKnn, y = YRes)$pred + runif(nrow(dataUnlabeled), min = 0, max = 1e-10)
     EcdfKnn <- ecdf(sigmaKnnPredUnlabeled)
  
##prediction on dataTest
     predT <- knn.reg(train  = as.matrix(dataTrain$X), test  = as.matrix(dataTest$X), k = paramKnn, y = YRes)$pred + runif(nrow(dataTest$X),min = 0, max= 1e-10)
     scoreKnn <-  EcdfKnn(predT)
     accept <- which(scoreKnn <= 1-eps) ##indices of the instances which is accepted by the procedure (e.g. not rejected)
     prediction <- knn.reg(train = as.matrix(dataTrain$X), test = as.matrix(dataTest$X), k = paramKnn, y = dataTrain$Y)$pred[accept] ## prediction of the accepted instances
     if(length(accept)==0){results[Nrep,1] = 0}else{results[Nrep,1] <- (mean((dataTest$Y[accept]-prediction)^2))} ##computation of the error rate
     results[Nrep, 2] <- 1- length(accept)/length(dataTest$Y) ##computation of the rejection rate
  }

cat("\n", "desired rejection =",eps, "\n")
cat("\n", "obtained results (mean)", "\n")
print(round(apply(results,2,mean), digits = 2))

cat("\n", "obtained results (sd)", "\n")
print(round(apply(results,2,sd), digits = 2))

























