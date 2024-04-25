//
// attempting Schnabel population estimate in Bayesian context

// for now this is for one lake only; can add nesting later

// The input data is a vector 'y' of length 'N'.
data {
  // number of samples
  int<lower=0> T;
  // vector of catch in each sample t
  //vector[T] Ct;
  // vector of marks at large prior to each sample t
  vector[T] Mt;
  // recaps at each sampling event t
  vector[T] Rt;
  
  //int<lower=0> sumRt;
}

// The parameters accepted by the model. 
parameters {
  // left off here
  
  // make bounds more specific (based on Mt Rt values)
 // real<lower=0> PE;
  real<lower=0> nUnmarked;
}

// The model to be estimated. 
model {
  sum(Rt)~hypergeometric(nUnmarked+sum(Mt), Mt[T], nUnmarked);
  
 // PE=nUnmarked[T]+sum(Mt);
  
  // this is very specific and maybe wrong?
  nUnmarked~gamma(0.1,0.1);
  


}


