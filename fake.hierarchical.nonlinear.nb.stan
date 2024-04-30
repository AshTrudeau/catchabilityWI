//
// fitting nonlinear catch equation to fake data

// The input data is a vector 'y' of length 'N'.
data {
  // number of observations
  int<lower=0> N;
  // number of groups (anglers)
  int<lower=1> A;
  // all observations of catch (response)
  array[N] int<lower=0> lmbCatch;
  array[N] int<lower=1, upper=A> AA;
  // all observations of effort (covariate)
  //vector<lower=0>[N] effort;
  array[N] real<lower=0> effort;
  // all estimates of population density (covariate)
  //vector<lower=0>[N] popDensity;
  array[N] real<lower=0> popDensity;
  
}


// phi is inverse overdispersion param
parameters {
  // I want different estimates of q for different anglers
  array[A] real<lower=0> q_a;
  // catch rates are generally hyperstable (so <1), but leaving some wiggle room to see if it still works
  real<lower=0, upper=2> beta;
  real<lower=0> phi;
  
  // hyperparameters
  real<lower=0> mulog_q;
  real<lower=0> sigmalog_q;
}

transformed parameters{

  vector[N] catchHat;
  vector[N] logCatchHat;
  
  for(i in 1:N){
    catchHat[i] = effort[i] .* q_a[AA[i]] .* popDensity[i]^beta;
  }
  
  logCatchHat = log(catchHat);


}
// use log alternative parameterization for NB? NegBinomial2Log
model {
  
  lmbCatch~neg_binomial_2_log(logCatchHat, phi);
  
  q_a~lognormal(mulog_q, sigmalog_q);
  
  mulog_q~normal(0, 10);
  sigmalog_q~cauchy(0,5);

  phi~gamma(0.001,0.001);

  beta~lognormal(0,1);
  
  
}

