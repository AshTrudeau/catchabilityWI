//
// attempting Schnabel population estimate in Bayesian context

// for now this is for one lake only; can add nesting later

// The input data is a vector 'y' of length 'N'.
data {
  // number of samples
  int<lower=0> T;
  // vector of catch in each sample t
  vector[T] Ct;
  // vector of marks at large prior to each sample t
  vector[T] Mt;
  // recaps at each sampling event t
  vector[T] Rt;
  
  int<lower=0> sumRt;
}

// The parameters accepted by the model. 
parameters {
  // left off here
  real<lower=0> inv_PE;
  real<lower=0> PE_a;
  real<lower=0> PE_b;
}

// The model to be estimated. 
model {
  sumRt~poisson((Ct*Mt)*(inv_PE));
  
  inv_PE~gamma(PE_a,PE_b);
  
  PE_a~cauchy(0,5);
  PE_b~cauchy(0,5);
  
}


