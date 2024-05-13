//
// fitting nonlinear catch equation to fake data

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  //int lmbCatch[N];
  array[N] int lmbCatch;
  vector<lower=0>[N] effort;
  vector<lower=0>[N] popDensity;
  
}


// phi is inverse overdispersion param
parameters {
  real<lower=0> q;
  real<lower=0> beta;
  real<lower=0> phi;
}

transformed parameters{
  
  vector[N] catchHat;
  vector[N] logCatchHat;
  
  catchHat = effort .* q .* popDensity^beta;
  
  logCatchHat = log(catchHat);


}
// use log alternative parameterization for NB? NegBinomial2Log
model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);

  phi~gamma(0.001,0.001);

  q~lognormal(0,1);
  beta~lognormal(0,1);
  
  
}

