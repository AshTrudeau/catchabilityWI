#Continuous Schnabel population estimate
#CTS 30 May 2018

#This is a quick and dirty version of Schnabel, based on Ricker 1975 (Computation and interpretation of biological
#statistics of fish populations) and Schabel 1938 (Am. Math. Mon.). It gives an approximation to the ML estimate of
#population size and the confidence limits are I think approximate too.
#Updated 30 July - wrote in option to use Poisson for CI when total number of recaps is low. Formula from Krebs,
#Ecological Methodology, in-prep edition posted on his web site. Should check.

schnabel <- function(markedPrior,collectedNow,recapturedNow) {
  #Function to calculate continuous Schnabel population estimate
  
  #Arguments:
  #markedPrior - Number of fish marked prior to each sampling event. Vector; first element is first sample, etc.
  #collectedNow - Number of fish collected in each sampling event, including fish that were previously marked and those that weren't. Vector; first element is first sample, etc.
  #recapturedNow - Number of recaptures in each sampling event. Vector; first element is first sample, etc.
  
  #Value:
  #A data.frame giving the point estimate and the lower and upper bounds of a 95% CI for each sampling event.
  
  #Rename arguments for convenience
  M <- markedPrior
  C <- collectedNow
  R <- recapturedNow
  
  #Number of sampling events
  nSamples <- length(markedPrior)
  
  #Set up output structure
  out <- data.frame(event=c(1:nSamples),nHatLow=rep(NA,nSamples),nHat=rep(NA,nSamples),nHatHigh=rep(NA,nSamples))
  
  #Do the calculations for each sampling event - point estimate and 95% CI
  for (i in 1:nSamples) {
    out$nHat[i] <- sum(M[1:i]*C[1:i])/sum(R[1:i])
    #If total number of recaps across all samples to date is > 50, use normal approximation CI
    if (sum(R[1:i])>50) {
      out$nHatLow[i] <- 1/(out$nHat[i]^-1+1.96*sqrt(sum(R[1:i]/sum(M[1:i]*C[1:i])^2)))
      out$nHatHigh[i] <- 1/(out$nHat[i]^-1-1.96*sqrt(sum(R[1:i]/sum(M[1:i]*C[1:i])^2)))
    } else
      #If total number of recaps across all samples to date is =< 50, use Poisson (check this method)
      if (sum(R[1:i])<=50) {
        out$nHatLow[i] <- sum(M[1:i]*C[1:i])/qpois(0.975,sum(R[1:i]))
        out$nHatHigh[i] <- sum(M[1:i]*C[1:i])/qpois(0.025,sum(R[1:i]))
      }
  }
  
  #Return result
  return(out)
  
}